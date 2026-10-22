# DR-006 — Bootstrap inference strategy for H1 and H2

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze), Issue #14, in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition |
| Date | 2026-09-24; revised 2026-10-09 (Issue #54: eight-configuration model set; bounded bootstrap p-values) |
| Evidence source | Development-period (V2, relative days 127–154) dependence diagnostics, computed for this decision with no Test labels and no model scores; `notebooks/01_dataset_feasibility_audit.ipynb` §15; bootstrap and clustered-inference literature |
| Related | [DR-001](DR-001-temporal-split-strategy.md), [DR-002](DR-002-review-capacity-definition.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-005](DR-005-sesoi-and-h1-h2-inference.md), [DR-011](DR-011-model-set-tree-based-and-neural-families.md), [protocol](../protocol.md) |

## Context

Confirmatory hypotheses H1 (model family) and H2 (class weighting) compare paired differences in pooled Test **Recall@2%** (share of frauds reviewed at 2 % review capacity) under the daily top-K policy R3 (review the K highest-scoring transactions each relative day; K = 56 reviews per relative day) across the eight model × weighting configurations of [DR-011](DR-011-model-set-tree-based-and-neural-families.md) ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)). DR-005 requires a **joint paired bootstrap** that does not assume independent transactions or frauds. The **resampling unit** has two natural candidates: heuristic pseudo-entity clusters and day blocks.

The trade-off:
- entity resampling targets within-entity dependence, but relies on a **heuristic** key;
- day resampling targets temporal dependence and keeps each day's top-K intact, but the Test period has only **28 days**.

This record fixes that choice using development-period evidence only.

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

- **S1 — Owen-corrected two-way variance** [REF-068]:

  $$
  \begin{aligned}
  \mathrm{SE}^2 &= V_{\mathrm{pigeonhole}} - 2 \cdot V_{\mathrm{naive}} \\
  \text{Wald interval} &= \hat{\delta} \pm z \cdot \mathrm{SE}
  \end{aligned}
  $$

  where:

  - `SE` = corrected standard error of the estimated difference;
  - `V_pigeonhole` = variance of the estimate across pigeonhole replicates;
  - `V_naive` = variance from a fraud-level independent and identically distributed (iid) bootstrap;
  - `δ̂` = point estimate of the paired difference (δ_F or δ_W,m);
  - `z` = standard-normal quantile for the interval level (e.g. 1.96 for 95 %).

  This removes the known over-count and shows how far conclusions depend on the primary method's conservatism. If the corrected variance is ≤ 0, S1 is reported as not estimable.
- **S2 — Day-level moving-block bootstrap, ℓ (block length) = 3 days.** Circular wrap is not used, to match H3 ([DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md)). This is a key-free check that preserves each day's top-K exactly. The same B and interval type as the primary are used.

**Robustness rule.**
- Confirmatory decisions (H1, H2) use the **primary method only**; there is no switching after results are seen.
- If S1 or S2 would lead to a different H1 or H2 conclusion, this is reported explicitly as sensitivity of the conclusion to dependence assumptions.

### Exact paired resampling procedure (primary)

Inputs from the one confirmatory Test evaluation, computed **once**:
- For each of the eight configurations m ∈ {LGB-u, Cat-u, RES-u, FTT-u, LGB-b, Cat-b, RES-b, FTT-b} (LightGBM, CatBoost, ResNet-style MLP or FT-Transformer, unweighted or balanced weighting; [DR-011](DR-011-model-set-tree-based-and-neural-families.md)) and each Test relative day d, rank that day's transactions by m's score. Set c_{m,i} = 1 if fraud i is among day(i)'s top K = 56 under m, else 0. Ties are handled as pre-declared in the implementation.
- For each Test fraud i: its day d(i), and its fine pseudo-entity e(i), built from Test-period features only (card1, addr1, P_emaildomain, relative day − D1).

For each bootstrap replicate b = 1, …, B (B = number of replicates):

1. Draw day multiplicities over the 28 Test days, and, independently, entity multiplicities over the G fraud entities in Test:

   $$
   \begin{aligned}
   (a_1, \dots, a_{28}) &\sim \operatorname{Multinomial}\left(28;\ \tfrac{1}{28}, \dots, \tfrac{1}{28}\right) \\
   (b_1, \dots, b_G) &\sim \operatorname{Multinomial}\left(G;\ \tfrac{1}{G}, \dots, \tfrac{1}{G}\right)
   \end{aligned}
   $$

   where:

   - `a_d` = number of times Test day d is drawn in this replicate;
   - `b_e` = number of times fraud entity e is drawn in this replicate;
   - `G` = number of distinct fine pseudo-entities among Test frauds.

2. Weight each fraud by the product of its day's and its entity's multiplicities:

   $$
   w_i = a_{d(i)} \cdot b_{e(i)}
   $$

   where `d(i)` and `e(i)` are the day and the entity of fraud i.

3. For **every** configuration m, using the **same** weights, compute the replicate recall:

   $$
   R^*_m = \frac{\sum_i w_i \cdot c_{m,i}}{\sum_i w_i}
   $$

   where `c_m,i` = 1 if fraud i is among its day's top K under configuration m, else 0 (the capture indicator), and `Σ_i` sums over Test frauds.

4. Compute the replicate estimands:

   $$
   \begin{aligned}
   \delta^*_F &= \frac{1}{4} \sum_{m \in N} \sum_{w} R^*_{m\text{-}w} \;-\; \frac{1}{4} \sum_{m \in T} \sum_{w} R^*_{m\text{-}w} \\
   \delta^*_{W,m} &= R^*_{m\text{-}b} - R^*_{m\text{-}u} \qquad \text{for } m \in \lbrace \mathrm{LGB}, \mathrm{Cat}, \mathrm{RES}, \mathrm{FTT} \rbrace
   \end{aligned}
   $$

   where δ*_F and δ*_W,m are the model-family effect and the weighting effect of model m, computed in this replicate; T = {LGB, Cat}, N = {RES, FTT} and w ∈ {u, b}. The descriptive contrasts and the interaction are computed in the same way.

**Why this is valid for paired comparison.**
- The weights depend only on each fraud's (day, entity) membership, never on model outputs.
- Applying the same weights to all eight configurations keeps the within-fraud correlation between configurations. The bootstrap distribution of each difference therefore reflects paired, not independent-sample, variability [REF-068, REF-036].

**Daily top-K: why the ranking is not recomputed in each replicate.**
- In the **primary**, recomputing top-K on a "resampled dataset" is not well defined. Duplicating an entity inside a day would create tied duplicate transactions and change that day's volume while K stays 56, so the capacity rate would no longer be 2 %.
- The primary therefore conditions on the realised within-day ranking. Uncertainty comes from *which days and which fraud entities occur*. Day-level variation in capacity competition is carried by the day weights, which move all of a day's frauds together.
- As a consequence, a replicate's weighted captured count for a day can exceed K. The replicate is a weighted estimator used for variance approximation, not a literal simulated Test period.
- In **S2**, whole days are resampled. Each day's top-K is then *exactly* preserved, and recomputation would give identical results, so none is needed.


### Confidence intervals and replicates

| Item | Specification | Justification |
|---|---|---|
| Interval type (primary) | **Percentile** confidence intervals (CIs) from the pigeonhole distribution [REF-044] | Simple, respects the non-normal shape of a bounded ratio, and matches H3's percentile intervals. Mildly conservative by construction [REF-068]. |
| BCa (bias-corrected and accelerated interval) | **Not used** | Its acceleration constant is normally estimated by a jackknife, and no standard jackknife exists for crossed two-way data. Using BCa here would add an unvalidated component. |
| Basic / studentized | Not used as primary | A studentized interval needs a per-replicate variance estimate, which would need a second-level two-way bootstrap. The added complexity is not justified for this confirmatory analysis. S1 provides a Wald-type check. |
| Levels | 95 % for difference estimates; 90 % for H1's practical interpretation and for the H2 TOST (Two One-Sided Tests) contrasts, reported descriptively. Confirmatory decisions use the bootstrap p-values with Holm adjustment | As DR-005 |
| Replicates | **B = 10,000**, fixed seed recorded | Holm adjustment compares the smallest p-value with 0.05 / 3 ≈ 0.0167, so tail proportions near 1.67 % must be estimated precisely: ≈ 167 replicates in the tail at B = 10,000, against ≈ 17 at B = 1,000, which is too few. Computation is negligible. Monte Carlo standard errors of reported p-values are also reported. |
| Small-cluster corrections | **None applied**; limitation stated | Wild cluster bootstrap-t corrections [REF-069] are designed for regression coefficients with few clusters in one dimension. Here the entity dimension is large (≈ 1,350 clusters projected), the day dimension (28) carries the few-cluster risk, and the primary's conservatism partly offsets it. |
| Day blocks | Primary: iid days inside the two-way scheme. Sensitivity S2: fixed ℓ = 3 | Within V2, lag-1 autocorrelation of daily prevalence is 0.01 [D]. Once entity recurrence is modelled explicitly, the remaining serial dependence is small. S2 keeps the H3 block length (DR-003; [REF-064]) for comparability. |


### Relationship to H1, H2, TOST and McNemar

- **H1 (difference).** The two-sided bootstrap p-value is the percentile-interval inversion, with the add-one correction and a cap at 1 ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)):

  $$
  p_{\mathrm{H1}} = \min\left\lbrace 1,\ 2 \cdot \min\left( \frac{1 + \#\lbrace \delta^*_F \le 0 \rbrace}{B + 1},\ \frac{1 + \#\lbrace \delta^*_F \ge 0 \rbrace}{B + 1} \right) \right\rbrace
  $$

  where `#{ · }` = the number of bootstrap replicates satisfying the condition. The 90 % percentile interval gives DR-005's five-way practical interpretation against ±0.02 (± the Smallest Effect Size Of Interest, SESOI).
- **H2 (intersection-union TOST).** For each model m, the two one-sided p-values and the H2 p-value are:

  $$
  \begin{aligned}
  p_{a,m} &= \frac{1 + \#\lbrace \delta^*_{W,m} \le -0.02 \rbrace}{B + 1} \\
  p_{b,m} &= \frac{1 + \#\lbrace \delta^*_{W,m} \ge +0.02 \rbrace}{B + 1} \\
  p_{\mathrm{H2}} &= \max_{m \in \lbrace \mathrm{LGB}, \mathrm{Cat}, \mathrm{RES}, \mathrm{FTT} \rbrace} \max\left(p_{a,m},\ p_{b,m}\right)
  \end{aligned}
  $$

  where p_a,m and p_b,m test the lower and upper equivalence bounds for model m. H2 needs all eight one-sided tests to reject, and is decided by the Holm-adjusted value of p_H2 [REF-061, REF-062] ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)). **TOST uses the bootstrap distribution**, because no closed-form interval accounts for the crossed dependence. The percentile intervals of δ_W,m are reported. They describe uncertainty only and do not determine rejection, because the decision uses the Holm-adjusted p-value.
- **Multiplicity.** Holm over {H1, H2, H3} ([DR-005](DR-005-sesoi-and-h1-h2-inference.md)).
- **H3** is label-free and uses the day-level moving-block bootstrap of [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md).
- **McNemar remains secondary and descriptive.** It assumes independent paired units, and its size "can greatly exceed the nominal level" under intracluster correlation [REF-071]. It is reported for the cell-level contrasts only, labelled as independence-based, and is excluded from the Holm family.

## Evidence

### Observed dependence in V2 [D]

Two heuristic pseudo-entity keys from the feasibility audit were used:
- **fine**: card1 + addr1 + purchaser email domain + (relative day − D1);
- **coarse**: card attributes + addr1.

Neither is a verified customer, card or account identifier.

| Quantity (V2, Validation 2 = relative days 127–154: 28 days, 77,822 transactions, 2,637 frauds) | Value |
|---|---|
| Fraud clusters, fine key | 1,353. 61.7 % are singletons; **68.3 % of frauds** are in clusters of ≥ 2; largest 29 |
| Upper-bound design effect (DE) for fraud-level recall, fine key (formula below the table) | **4.25** |
| V2 frauds whose fine-key cluster spans more than one day | **44.7 %** |
| Same-cluster fraud pairs, by day gap (fine key) | same day 29.1 %; 1–2 days 21.4 %; 3–6 days 18.9 %; **≥ 7 days 30.7 %** |
| Fine-key cluster span (last − first fraud day), multi-fraud clusters | median 1 day, 90th percentile 14 days |
| Daily fraud-count dispersion (variance ÷ binomial variance) | **5.3** in V2 (8.6 over days 1–154) |
| Lag-1 autocorrelation of daily prevalence | **0.01** in V2 (0.39 over days 1–154, which includes between-period level shifts) |
| Every V2 day has more frauds than K = 56 | Yes, all 28 days (capacity binds daily; [DR-002](DR-002-review-capacity-definition.md)) |
| Coarse key | 735 clusters; upper-bound DE 15.2; 76.6 % of frauds in clusters spanning > 1 day |

The upper-bound design effect is:

$$
\mathrm{DE} = \frac{\sum_g m_g^2}{\sum_g m_g}
$$

where:

- `g` = a fraud cluster (pseudo-entity);
- `m_g` = number of frauds in cluster g;
- `DE` = design effect: how much clustering can inflate the variance of fraud-level recall relative to independent frauds.

The more frauds concentrate in large clusters, the larger DE becomes; DE = 1 when every fraud is its own cluster.

### Proxy variance diagnostic [D, with interpretation I]

**Why a proxy is needed.** Model capture indicators do not exist yet: no models have been trained and no Test scores exist. As a proxy, four **entity-linked binary attributes of V2 frauds** were used: identity present; ProductCD = C; card6 = credit; amount above the V2 fraud median.

**Method.** For each attribute, the variance of the fraud-level proportion was estimated under several resampling schemes (4,000 replicates each, fixed seed). Each variance is expressed relative to a naive fraud-level bootstrap.

| Attribute (fine key) | Day (iid days: days resampled as independent and identically distributed) | Day blocks, ℓ (block length) = 3 days | Entity (fine key) | Two-way pigeonhole |
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
  - For crossed, unbalanced random-effects data, **resampling rows and columns independently** (the pigeonhole bootstrap) is mean-consistent. It is **mildly conservative**, as the variance approximation of Owen (2007) [REF-068] shows:

    $$
    \begin{aligned}
    V_{\mathrm{pigeonhole}} &\approx \frac{(\nu_A + 2)\,\sigma_A^2 + (\nu_B + 2)\,\sigma_B^2 + 3\,\sigma_E^2}{n_{\mathrm{obs}}} \\[6pt]
    \text{true variance} &= \frac{\nu_A\,\sigma_A^2 + \nu_B\,\sigma_B^2 + \sigma_E^2}{n_{\mathrm{obs}}}
    \end{aligned}
    $$

    where:

    - `V_pigeonhole` = the pigeonhole bootstrap's variance estimate;
    - `A`, `B` = the two crossed factors (here: relative day and pseudo-entity);
    - `σ²_A`, `σ²_B` = variance components of factors A and B; `σ²_E` = residual variance;
    - `ν_A`, `ν_B` = Owen's imbalance measures for factors A and B, as defined in REF-068;
    - `n_obs` = number of observations.

    The pigeonhole estimate therefore overestimates by about twice the naive variance, and Owen suggests subtracting twice the naive variance as a correction. A naive case bootstrap "can be seriously misleading" [REF-068].
- **Few clusters.** With few (five to thirty) clusters, standard cluster-robust tests can over-reject [REF-069]. With 28 Test days, this applies to any day-only method.
- **Temporal dependence.** Block bootstraps handle short-range serial dependence [REF-037]; block-length orders are given by [REF-064]. They cannot absorb dependence at lags comparable to the series length.
- **Paired binary comparisons under clustering.** "In the presence of intracluster correlation, the size of McNemar's test can greatly exceed the nominal level" [REF-071]. Paired classifier comparisons that assume independent observations [REF-028, REF-035] are therefore not valid primary tests here.
- **No fraud-specific standard.** No verified source prescribes a bootstrap design for capacity-constrained recall in fraud detection. The choice in [Decision](#decision) is an application of the general methods above.

## Alternatives considered

| | A. Transaction-level | B. Pseudo-entity cluster (one-way) | C. Day-level, iid days | D. Moving-block days (ℓ = 3) | **E. Two-way pigeonhole (day × fine entity)** |
|---|---|---|---|---|---|
| Dependence handled | None | Within-entity, including across days | Within-day, including same-day entity clustering and daily capacity competition | Within-day plus short-range (≤ ≈ 3 days) serial dependence | **Both** day-level and entity-level, including cross-day entity recurrence |
| Dependence ignored | All | Day-level shocks and capacity competition shared across entities | Cross-day entity recurrence (45 % of frauds) | Entity recurrence beyond ≈ 3 days (≈ half of same-entity pairs) | Residual dependence not captured by the heuristic key; serial dependence between days (small in V2) |
| Key assumptions | Independence (contradicted) | The key captures entities; days exchangeable | Days exchangeable | Short-range stationarity | Crossed random-effects structure; iid days; key approximately captures entities |
| Recall@2% compatibility | Yes | Yes, via fixed capture indicators | Yes, exactly (days intact) | Yes, exactly | Yes, via fixed capture indicators |
| Paired comparison | Yes | Yes | Yes | Yes | Yes |
| Preserves daily top-K | No | Conditionally (rankings fixed) | **Exactly** | **Exactly** | Conditionally (rankings fixed; day dimension carries day-level variation) |
| Feasible with 28 days | — | Not day-dependent (≈ 1,350 entity clusters) | Few-cluster risk [REF-069] | Worse: ≈ 26 distinct blocks, ≈ 10 per replicate | Day dimension carries the few-cluster risk; entity dimension large |
| Dependence on heuristic key | None | **Total** | None | None | Partial: if the key fails, E degrades towards a day + fraud-level bootstrap, which is still conservative relative to C |
| Computation | Trivial | Trivial | Trivial | Trivial | Trivial (product weights on ≈ 2,700 frauds) |
| Interpretability | High but wrong | Moderate | High | Moderate | Moderate |
| Verdict | **Rejected** | Sensitivity input only (via E) | Rejected as primary: misses cross-day entity dependence | **Sensitivity (key-free)** | **Primary** |

## Consequences

- The Test evaluation saves, for every one of the eight configurations: per-fraud capture indicators; each fraud's relative day and fine pseudo-entity; per-day TP (true positives: frauds reviewed) and fraud counts (for S2).
- **Power may be lower than DR-005's central scenario.** The proxy diagnostic suggests two-way variance ratios of ≈ 5–10 times naive for entity-linked quantities, whereas DR-005 assumed design effects of 1–4. Paired differences may be less affected, because shared entity effects cancel, but inconclusive H1/H2 outcomes are **more** likely than DR-005's central scenario indicates. This is recorded so that inconclusive results are interpreted correctly.
- The same fine pseudo-entity definition is used for the bootstrap and for the "recurring vs new entity" false-negative slice. Both are heuristic.

## Threats and limitations

- **The fine pseudo-entity key is heuristic.** If it splits true entities, residual cross-day dependence is not captured. If it merges unrelated ones, variance is overstated.
- **The primary method is conservative by design.** It reduces power, particularly for H2 equivalence. S1 quantifies how much.
- **Conditioning on the realised rankings** means variability from the legitimate-transaction background within a day is represented only through the day weights, not by re-ranking. S2 re-ranks nothing either, but keeps whole days intact.
- **Only 28 days in the day dimension.** Coverage may fall slightly below nominal [REF-069]. This is partly offset by the conservative primary.
- **The proxy diagnostic** uses fraud attributes, not model capture indicators. It informs the choice of dimensions, not the exact size of the Test variance.
- **Parts of the literature were read only in part.** REF-069 and REF-070 were read at abstract level. REF-068's conservativeness and correction results were read in full text (arXiv version).

## References

REF-028, REF-035, REF-036, REF-037, REF-044, REF-061, REF-062, REF-064, REF-068, REF-069, REF-070, REF-071 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature, inference |
| A, B (factors) | The two crossed factors in Owen's model (here relative day and pseudo-entity) |
| a_d | Multiplicity of Test day d in a bootstrap replicate |
| B | Number of bootstrap replicates (10,000) |
| b | Index of a bootstrap replicate, b = 1, …, B |
| b_e | Multiplicity of fraud entity e in a bootstrap replicate |
| BCa | Bias-corrected and accelerated bootstrap interval |
| c_m,i | Capture indicator: 1 if fraud i is in its day's top K under configuration m, else 0 |
| Cat, LGB, RES, FTT; u, b (weighting) | CatBoost, LightGBM, ResNet-style MLP, FT-Transformer; unweighted, balanced weighting |
| CI | Confidence interval |
| d, d(i) | A Test relative day; the day of fraud i |
| DE | Design effect: variance inflation due to clustering |
| δ̂ | Point estimate of a paired difference |
| w | Class weighting, u or b (in the estimand formulas) |
| δ_F, δ*_F | Model-family effect (neural mean minus tree mean); its value in one bootstrap replicate |
| δ_W,m, δ*_W,m | Weighting effect of model m; its value in one bootstrap replicate |
| e, e(i) | A fraud pseudo-entity; the entity of fraud i |
| G | Number of distinct fine pseudo-entities among Test frauds |
| g, m_g | A fraud cluster; number of frauds in it |
| H1, H2, H3 | Confirmatory hypotheses: model family; class weighting; alert-rate transfer |
| iid | Independent and identically distributed |
| K | Fixed daily review budget (56 at 2 % capacity) |
| ℓ | Block length of a moving-block bootstrap, in days |
| m | A model × weighting configuration (in the H2 formulas: a model) |
| MLP | Multilayer perceptron |
| N, T | Neural family {RES, FTT}; tree-based family {LGB, Cat} |
| #{ · } | Number of bootstrap replicates satisfying a condition |
| n_obs | Number of observations (in the variance approximation of Owen, REF-068) |
| ν_A, ν_B | Owen's imbalance measures for factors A and B, as defined in REF-068 |
| p_a,m, p_b,m | One-sided H2 p-values for the lower and upper bounds of model m |
| p_H1, p_H2 | p-values of H1 and H2 |
| R3 | Decision policy: daily top-K review |
| Recall@2% | Share of frauds reviewed at the 2 % review capacity |
| S1, S2 | Sensitivity analyses: Owen-corrected variance; day-level moving-block bootstrap |
| SE | Standard error |
| SESOI | Smallest Effect Size Of Interest |
| σ²_A, σ²_B, σ²_E | Variance components of factors A and B; residual variance |
| TOST | Two One-Sided Tests |
| TP | True positives: frauds reviewed (per day, for S2) |
| V_naive, V_pigeonhole | Variance from a fraud-level iid bootstrap; variance across pigeonhole replicates |
| V2 | Validation 2 (relative days 127–154) |
| w_i | Bootstrap weight of fraud i |
| z | Standard-normal quantile for the interval level |
