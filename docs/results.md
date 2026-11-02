# Confirmatory Results

The interpretation of the single confirmatory Test evaluation of the study: run `2026-11-01T104408Z`, protocol frozen at the Git tag `protocol-freeze-v1`. The numbers are in the [promoted evidence](../evidence/confirmatory/README.md) ([`summary.md`](../evidence/confirmatory/2026-11-01T104408Z/summary.md) and the JSON (JavaScript Object Notation) records); this document selects from them and reads them. The design, the estimands and the decision rules are owned by the [protocol](protocol.md) and the [Decision Records](decisions/README.md) (DR). The pre-registered decisions are reported as recorded and nothing here changes a decision, threshold or hypothesis.

Evidence tags: **[D]** observed in the evidence; **[L]** literature (`REF-NNN`, [references](references.md)); **[I]** interpretation of this document. The study asks three research questions (RQ1 to RQ3) and tests three hypotheses (H1 to H3), all defined in the [protocol](protocol.md); the conventions of the intervals and of the positions against the equivalence band (beyond, inside, crossing) are those of [DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md) and [DR-006](decisions/DR-006-bootstrap-inference-strategy.md).

## Notation used in the tables

- **pp** percentage points; **K** the fixed daily review budget (reviews per relative day), 56 at the 2 % capacity; **Recall@2 %** the share of the Test frauds reviewed when the daily top-K review (policy R3) fills K = 56.
- A **configuration** is `<model>-<weighting>`: LGB (LightGBM), Cat (CatBoost), RES (ResNet-style multilayer perceptron, MLP), FTT (FT-Transformer, Feature Tokenizer + Transformer); `-u` unweighted, `-b` balanced class weighting; **LR** is the logistic-regression reference ([DR-011](decisions/DR-011-model-set-tree-based-and-neural-families.md)).
- **R2** is the capacity-rate threshold policy and **R4** the calibrated cost-aware capped policy; **ρ** is the miss-to-review cost ratio, **ρ_sat** the saturation ratio from which R4 fills the budget, and **ΔJ** the scenario-cost difference of R4 from R3, defined in the RQ3 section.
- The **equivalence band** is ±SESOI (Smallest Effect Size Of Interest, 0.02 of Recall@2 %); an interval is *beyond* it, *inside* it or *crossing* a bound of it ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md)).

## Answers at a glance

| Question | Decision as recorded [D] | Answer [I] |
|---|---|---|
| **RQ1, H1**: do tree-based and neural models differ at a fixed review budget? | H1 rejected (Holm-adjusted p = 0.0006); effect −6.51 pp, beyond the band | Yes: the tree-based models review more frauds than the neural models, by a practically meaningful margin |
| **RQ1, H2**: is the effect of class weighting within each model practically negligible (equivalent)? | Equivalence not established (Holm-adjusted p = 0.687) | Not shown; CatBoost alone has a lower recall with weighting (−2.60 pp, descriptive) |
| **RQ2, H3**: does a validation-chosen threshold keep its alert rate later? | H3 preserved for LGB-b (Holm-adjusted p = 0.0284); rate 2.14 % | Yes for the pooled rate (within ±20 % of 2 %); the daily load exceeds the budget on 15 of 28 days |
| **RQ3**: when is a cost-aware policy cheaper than filling the budget? | Estimation, no confirmatory hypothesis | Below the saturation ratio ρ_sat (3.1 to 26.5), growing with the capacity; interval-based cells show it from the 2 % budget, none at 1 % |
| **Main question**: how do family, weighting and policy affect capture, workload and cost? | Answered through RQ1 to RQ3 | A practically meaningful family difference was found; weighting equivalence remained unresolved; LGB-b preserved its pooled alert rate without meeting the daily budget; the scenario-cost advantage of R4 depended on the ratio, the capacity and the calibration |

**For the full evidence**, follow the links of each section to the generated summary, the records and the figures; the design is in the [protocol](protocol.md).

## Key results

The tables select from the [secondary outcomes](../evidence/confirmatory/2026-11-01T104408Z/summary.md#secondary-outcomes), [H2 by model](../evidence/confirmatory/2026-11-01T104408Z/summary.md#h2-by-model), [H3 variants](../evidence/confirmatory/2026-11-01T104408Z/summary.md#h3-variants-and-tolerance-views), [RQ2](../evidence/confirmatory/2026-11-01T104408Z/summary.md#rq2-daily-budget-fidelity-of-r2-point-values) and [RQ3](../evidence/confirmatory/2026-11-01T104408Z/summary.md#rq3-r4-against-r3) tables of the summary, which hold the full set.

**Capture at the 2 % budget (R3, K = 56 reviews per day)** [D]. Recall@2 % is the share of the 2,774 Test frauds reviewed; normalised recall is that share divided by the oracle ceiling of 56.45 % (a perfect ranker, [DR-002](decisions/DR-002-review-capacity-definition.md)).

| Configuration | Recall@2 % | 95 % interval | Normalised recall | Missed frauds |
|---|---|---|---|---|
| LGB-u | 0.378 | [0.332, 0.427] | 0.670 | 1,725 |
| Cat-u | 0.386 | [0.340, 0.435] | 0.684 | 1,703 |
| RES-u | 0.293 | [0.246, 0.342] | 0.519 | 1,961 |
| FTT-u | 0.329 | [0.283, 0.377] | 0.582 | 1,862 |
| LGB-b (the H3 configuration) | 0.385 | [0.340, 0.433] | 0.683 | 1,705 |
| Cat-b | 0.360 | [0.316, 0.406] | 0.638 | 1,775 |
| RES-b | 0.300 | [0.260, 0.342] | 0.531 | 1,943 |
| FTT-b | 0.328 | [0.282, 0.376] | 0.581 | 1,864 |
| LR (reference) | 0.309 | [0.269, 0.352] | 0.548 | 1,916 |

**Weighting effect by model (H2, balanced minus unweighted Recall@2 %)** [D]. The band is ±2 pp.

| Model | Effect | 90 % interval | Position | Exact McNemar p (independence-based, descriptive) |
|---|---|---|---|---|
| LGB | +0.72 pp | [−0.84, +2.29] | crossing | 0.145 |
| Cat | −2.60 pp | [−4.67, −0.62] | crossing | 0.00007 |
| RES | +0.65 pp | [−2.46, +3.96] | crossing | 0.376 |
| FTT | −0.07 pp | [−2.27, +2.18] | crossing | 0.954 |

**Alert rate of the R2 threshold on Test (H3 target 2 %, band 1.6 % to 2.4 %)** [D].

| Configuration | Alert rate | 90 % interval | Position | Days above K = 56 (of 28) |
|---|---|---|---|---|
| LGB-b (confirmatory) | 2.136 % | [1.886 %, 2.332 %] | inside | 15 |
| LGB-u | 2.100 % | [1.866 %, 2.279 %] | inside | 14 |
| Cat-b | 2.164 % | [2.001 %, 2.357 %] | inside | 14 |
| FTT-u | 2.159 % | [1.942 %, 2.349 %] | inside | 16 |
| Cat-u | 2.250 % | [2.001 %, 2.452 %] | crossing | 17 |
| RES-b | 2.203 % | [1.996 %, 2.411 %] | crossing | 14 |
| FTT-b | 2.366 % | [2.162 %, 2.529 %] | crossing | 19 |
| RES-u | 3.023 % | [2.350 %, 4.058 %] | crossing | 20 |

**R4 against R3, Platt calibrator (8 configurations per capacity)** [D]. The regimes are those of [DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md): R4 cheaper or R3 cheaper when the 95 % interval of ΔJ lies on one side of zero, inconclusive when it includes zero, identical when R4 declines no queue member.

| Capacity | Saturation ratio ρ_sat | Break-even equals ρ_sat | At ρ = 2 | At ρ = 5 | At ρ = 10 |
|---|---|---|---|---|---|
| 1 % (K = 28) | 3.1 to 4.7 | 7 of 8 | 7 inconclusive, 1 R3 cheaper | 8 identical | 8 identical |
| 2 % (K = 56) | 5.0 to 8.2 | 7 of 8 | 6 R4 cheaper, 2 inconclusive | 7 inconclusive, 1 identical | 8 identical |
| 5 % (K = 139) | 9.2 to 14.9 | 8 of 8 | 8 R4 cheaper | 8 R4 cheaper | 7 inconclusive, 1 identical |
| 10 % (K = 278) | 14.2 to 26.5 | 7 of 8 | 8 R4 cheaper | 8 R4 cheaper | 6 R4 cheaper, 2 inconclusive |

## H1: tree-based against neural models

**Result [D].** The family effect δ_F (neural minus tree-based Recall@2 %, averaged over the two models of each family and over both weightings) is **−6.51 pp**, 90 % interval [−8.32, −4.75] pp and 95 % difference interval [−8.70, −4.39] pp. The two-sided bootstrap p is 0.0002 and the Holm-adjusted p is 0.0006 (the adjusted p-value decides; the interval positions describe), so H1 is rejected. The 90 % interval lies **beyond** the equivalence band, a difference of practically meaningful size by the protocol. The neural family reviews 180.5 fewer frauds per 28 days (6.4 per day) ([primary results](../evidence/confirmatory/2026-11-01T104408Z/summary.md#primary-confirmatory-results)).

**Robustness [D].** The Owen-corrected Wald variant (S1, a variance-based interval, not a percentile one) gives p = 1.3·10⁻¹³ and [−7.95, −5.06] pp; the day-level moving block (S2) gives p = 0.0002 and [−7.80, −5.46] pp; both are beyond the band with the same decision ([sensitivities](../evidence/confirmatory/2026-11-01T104408Z/summary.md#sensitivities)). The family effect stays beyond the band at SESOI (Smallest Effect Size Of Interest) 0.01 and 0.03 and crosses it at 0.05 (category "difference of uncertain practical size", a descriptive view outside the decision). Unweighted the effect is −7.12 pp ([−10.03, −4.41]) and balanced −5.89 pp ([−8.18, −3.68]); within the tree-based family CatBoost minus LightGBM is −0.87 pp ([−2.83, 1.11]) and within the neural family FT-Transformer minus ResNet-style multilayer perceptron (MLP) is +3.21 pp ([−0.02, 6.48]) ([descriptive estimands](../evidence/confirmatory/2026-11-01T104408Z/summary.md#descriptive-estimands-95--intervals)).

**Reading [I].** On this Test period the tree-based models capture more frauds at the 2 % budget, by a margin the pre-registered band classes as practically meaningful. The conclusion concerns the four models of the design under the common training procedure of [DR-015](decisions/DR-015-common-training-rules.md), not tree-based and neural models in general. The ranking measures agree in direction: average precision 0.47 to 0.51 for the tree-based and 0.28 to 0.41 for the neural configurations, and ROC-AUC (area under the receiver operating characteristic curve) 0.87 to 0.89 against 0.80 to 0.81 ([ranking measures](../evidence/confirmatory/2026-11-01T104408Z/summary.md#ranking-measures-day-level-moving-block)). The logistic-regression reference (LR) lies above the ResNet-style MLP configurations and below the tree-based and FT-Transformer configurations on Recall@2 %.

## H2: class weighting

**Result [D].** Equivalence of the weighting effect to within ±SESOI (0.02, 2 pp) is **not established**: the Holm-adjusted p is 0.687. The 90 % interval of the effect crosses a band bound for every model (table above). Neither dependence sensitivity changes this: p = 0.765 under S1 and 0.971 under S2.

**Reading [I].** "Not equivalent" is not "different": the data do not show the weighting effect to be smaller than 2 pp, and the intervals of LGB, RES and FTT are centred near zero but too wide to settle it (those of RES and FTT are the widest). CatBoost is the one model whose negative effect the 95 % interval excludes from zero ([−5.11, −0.23] pp; the paired McNemar test, which assumes independent frauds and is descriptive, finds 124 frauds only with balanced weighting and 196 only without it), while no effect larger than 2 pp is demonstrated either. At SESOI 0.03 the LGB and FTT intervals lie inside the band and at 0.05 all four do. Neither "weighting does not matter" nor "weighting harms" follows from the confirmatory decision; the CatBoost contrast is descriptive, outside the Holm family, from one Test period.

## H3: transfer of the alert rate (RQ2)

**Result [D].** The R2 threshold of LGB-b (LightGBM with balanced class weighting, the H3 configuration selected on Validation 2, V2, [DR-010](decisions/DR-010-h3-confirmatory-configuration.md#selection-result)) was set on V2 to flag 2 % of the transactions. On Test it flags **2.136 %**; the band is 2 % ± 20 % (±0.4 pp). The interval is **inside**, p = 0.0142 and Holm-adjusted p = 0.0284, so H3 is preserved. The day-block lengths 1 and 7 agree; the relative tolerances 0.25 and 0.50 give inside, 0.10 gives crossing, and t_min ≈ 0.166057 is the infimum of the relative tolerances whose open band contains the interval (containment requires a tolerance above the unrounded infimum) ([H3 variants](../evidence/confirmatory/2026-11-01T104408Z/summary.md#h3-variants-and-tolerance-views)).

**Daily workload [D].** LGB-b flags 59.9 transactions a day on average against K = 56 (load ratio 1.07): above the budget on 15 of 28 days (276 alerts above it), below it on 13 days (166 unused slots), with a mean absolute deviation of 15.8 alerts and no day without alerts. The other configurations range from 58.9 alerts a day (LGB-u) to 84.8 (RES-u, 20 days above, 903 alerts above the budget) ([RQ2 tables](../evidence/confirmatory/2026-11-01T104408Z/summary.md#rq2-daily-budget-fidelity-of-r2-point-values)).

**Reading [I].** A validation-chosen capacity-rate threshold keeps its intended pooled alert rate within the pre-registered tolerance for the H3 configuration. For the other configurations the intervals of LGB-u, Cat-b and FTT-u also lie inside the band and the others cross it; those positions are descriptive, and the preservation conclusion is reserved for the confirmatory configuration ([DR-010](decisions/DR-010-h3-confirmatory-configuration.md)). The result is about the pooled rate. It does not establish that a fixed threshold meets a fixed daily staffing budget: the same threshold exceeds K on more than half of the days, which is why the daily-budget outcomes are reported separately ([DR-009](decisions/DR-009-overflow-handling-for-threshold-policies.md), [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)).

## RQ3: when a cost-aware policy is cheaper than filling the budget

**Cost rule [I].** The scenario cost of a policy is

$$
J = N_{\mathrm{rev}} + \rho \cdot \mathrm{FN}
$$

where:

- `J` = scenario cost of the policy;
- `N_rev` = number of reviews;
- `ρ` = miss-to-review cost ratio (a scenario assumption, never an institutional cost);
- `FN` = number of missed frauds.

R3 fills the daily budget; R4 reviews only the queue members whose calibrated probability reaches the cost-derived threshold p*, and R4 is cheaper when ΔJ is negative ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)):

$$
p^* = \frac{1}{\rho}, \qquad \Delta J = J_{\mathrm{R4}} - J_{\mathrm{R3}}
$$

where:

- `p*` = cost-derived probability threshold of R4;
- `J_R3`, `J_R4` = scenario cost of R3 and of R4;
- `ΔJ` = their difference (R4 cheaper when negative).
 The results are conditional on the frozen models and calibrators and on the assumed costs; where calibration is poor they are the joint effect of the cost rule and calibration transfer, not evidence for or against cost-aware reviewing as such.

**Where R4 differs from R3 [D].** With the Platt (primary) calibrator the regimes at the grid ratios are, over the 32 configuration–capacity cells: at ρ = 2, R4 cheaper in 22, inconclusive in 9 and **R3 cheaper in 1** (RES-b at 1 %: ΔJ = +56, 95 % interval [5, 107]); at ρ = 5, R4 cheaper in 16, inconclusive in 7 and identical in 9; at ρ = 10, R4 cheaper in 6, inconclusive in 9 and identical in 17; at ρ = 20, identical in 26 and inconclusive in 6; and for ρ ≥ 50 identical in all 32 (R4 declines no queue member). The advantage of R4 appears below the saturation ratio ρ_sat (the ratio from which R4 fills the budget, so that it merges with R3), which grows with the capacity (table above): no 1 % cell is classified R4 cheaper at ρ = 2 on the tabulated grid. At the primary 2 % capacity LGB-b at ρ = 2 leaves 404 of 1,568 reviews unused and forgoes 144 frauds, for ΔJ = −116 cost units (−2.3 % of J_R3); at ρ = 5 R4 declines 4 queue members and the difference is inconclusive. The dataset's reference ratio, the inverse of the V2 fraud prevalence, is 29.5, above every Platt ρ_sat (at most 26.5) ([RQ3](../evidence/confirmatory/2026-11-01T104408Z/summary.md#rq3-r4-against-r3)).

**Break-even [D].** For Platt the break-even ratio ρ_BE coincides with ρ_sat (R4 is cheaper right up to the merging ratio, the pattern expected under good calibration) in 29 of 32 pairs and is a crossing (ρ_BE < ρ_sat: R4 becomes dearer before the merging) in three: FTT-b at 1 % (3.32 against 3.90), RES-b at 2 % (6.08 against 6.28) and LGB-u at 10 % (19.29 against 24.16).

**Declined-set diagnostic [D].** The observed fraud rate of the declined queue members against their mean calibrated probability and p* varies by cell. LGB-b at 2 % and ρ = 2 declines 404 members at an observed rate of 0.356 against a mean probability of 0.365 (p* = 0.5), and Cat-u at 2 % declines 371 at 0.364 against 0.374; RES-u at 2 % declines 555 at 0.328 against 0.324. At 1 % and ρ = 2 the rates exceed the mean probability: RES-b 0.561 against 0.370 (R3 cheaper, above) and FTT-b 0.590 against 0.412 (inconclusive, ΔJ = +64, [−1, 122]), both above the p* = 0.5 at which declining stops paying.

**Calibration sensitivity [D].** With isotonic regression 10 pairs are crossings (22 coincidences), 11 of the 32 pairs change classification and 8 of the 288 grid cells change regime; Cat-u, for example, changes from coincidence to crossing at 2 %, 5 % and 10 % ([calibration sensitivity](../evidence/confirmatory/2026-11-01T104408Z/summary.md#calibration-sensitivity-isotonic-against-platt)). Primary interpretation uses Platt only.

**Non-monotone curves [D].** The set of ratios where R4 is cheaper is not a single interval in 27 of 32 Platt results and 11 of 32 isotonic results; for these the cheaper intervals and the zero crossings are in the record and the result is read from the whole curve, never from the classification alone ([non-monotone curves](../evidence/confirmatory/2026-11-01T104408Z/summary.md#non-monotone-rq3-curves)).

**Reading [I].** Under the declared cost ratios, cost-aware capping lowers the scenario cost below ρ_sat, and the interval-based cells show this clearly from the 2 % capacity; from ρ_sat upwards R4 equals R3, so that filling the budget is as cheap as the cost-aware rule once a missed fraud costs more than about 3 to 26 reviews, depending on capacity and configuration. No 1 % cell is classified R4 cheaper on the tabulated grid (its intervals include zero), although the continuous point curves contain cheaper intervals below ρ_sat at 1 % (for example Cat-u for 1 < ρ < 3.30), so that the 1 % advantage is small against its uncertainty. The declined-set diagnostics show that R4's outcome depends on the calibration in the marginal region: where the declined rate matches the mean probability the decline is cost-correct, and where the declined group is richer in frauds than its probability (RES-b and FTT-b at 1 %) the cost advantage is absent or reversed. These outcomes are therefore read as the joint effect of the cost rule and calibration transfer, with the finite-sample uncertainty of the intervals retained. A second calibrator changes 11 of 32 classifications, so the exact break-even ratio is calibration-dependent and its order across configurations is not a model ranking.

## Calibration on the Test rows

**Result [D].** The diagnostics are those of logistic recalibration on the Test rows themselves (intercept 0 and slope 1 for perfect calibration, [REF-018](references.md#ref-018)) with the Brier score and the log loss. With the Platt calibrator the intercept intervals of all eight configurations include zero (RES-u: −0.30, [−0.88, 0.02]); the slopes are 0.93 to 0.98 for seven configurations (LGB-b 0.939, [0.888, 0.994]) and **0.66** for RES-u ([0.49, 0.89]). The Brier score is lowest for LGB-b (0.0227) and 0.0318 for RES-u ([calibration diagnostics](../evidence/confirmatory/2026-11-01T104408Z/summary.md#calibration-diagnostics-on-the-test-rows)). With isotonic regression the log-loss computation clips 2,156 probabilities for Cat-b and 15 to 209 for the other configurations; Platt clips none. In the reliability diagrams of the equal-count bins ([Platt](../evidence/confirmatory/2026-11-01T104408Z/figures/reliability_platt.svg), [isotonic](../evidence/confirmatory/2026-11-01T104408Z/figures/reliability_isotonic.svg)) the tree-based configurations follow the diagonal under both calibrators. Under Platt RES-u lies below it in its highest bin (mean probability about 0.44, observed rate about 0.30) and RES-b and FTT-b lie above it in theirs; under isotonic the highest bins of RES-b (0.375 predicted, 0.333 observed) and FTT-b (0.407, 0.365) lie below it.

**Reading [I].** A slope below 1 means that the calibrated probabilities are more spread than the outcomes support (overconfidence) [L, REF-018]. It is mild for the tree-based and FT-Transformer configurations, including LGB-b (whose slope interval, [0.888, 0.994], lies entirely below 1), and marked for RES-u. Global calibration and the calibration that R4 uses differ: RES-u has the weakest global calibration yet a close declined-set match at 2 %, whereas FTT-b has a mild global slope and a clear declined-set gap at 1 %. This is why the RQ3 outcomes are read as the joint effect of the cost rule and calibration transfer, and why a material-miscalibration label is not assigned by the software ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)). The calibration diagnostics do not alter the frozen rankings, the frozen thresholds or the confirmatory decisions.

## False-negative slices (exploratory)

**Result [D].** For LGB-b at the 2 % capacity, 1,705 of the 2,774 frauds are missed (61.5 %); the oracle ceiling of a perfect ranker at K = 56 is 56.45 % recall and LGB-b reaches 68.3 % of it (normalised recall 0.683). The miss rate is 85.9 % for frauds without identity information (1,191 frauds) against 43.1 % with it (1,583), 85.7 % when the device type is missing against 38.2 % on desktop and 45.6 % on mobile, and 87.0 % for product code W against 39.6 % for C. It rises with the amount band (48.0 % in the lowest, 71.4 % in the highest). The four weekly blocks are similar (57.7 % to 64.2 %). A fraud is *recurring* when its pseudo-entity has at least one fraud label in Train and *new* otherwise ([protocol](protocol.md#false-negative-analysis)): new accounts for 91.6 % of the misses (miss rate 62.8 %) and recurring for 8.4 % (50.0 %, interval [33.3 %, 66.4 %]). Of the missed frauds, 52.8 % have a within-day score rank greater than 5K ([false-negative slices](../evidence/confirmatory/2026-11-01T104408Z/summary.md#false-negative-slices-r3-at-the-primary-capacity)).

**Reading [I].** The analysis is exploratory and non-causal: slice base rates differ strongly and the slices overlap, so the miss rates describe where the review budget fails to reach frauds, not why. Frauds without identity or device records are missed far more often; whether this reflects less information in the model inputs or a different kind of fraud cannot be read from this analysis. Slices with fewer than the minimum number of frauds carry no interval.

## Computational cost

**[D]** The Test scoring of the 78,542 transactions took 0.4 to 3.0 s for the tree-based models and the reference, 1.8 to 1.9 s for the ResNet-style MLP and 46 s (FTT-b) and 327 s (FTT-u) for the FT-Transformer ([scoring time](../evidence/confirmatory/2026-11-01T104408Z/summary.md#test-scoring-time)). The timings are taken on different devices (central processing unit for the tree-based models, graphics processing unit for the neural models) and are not a same-device comparison.

## Limits of the evidence

The following boundaries define what the results support.

- **One Test period.** The 28 days of one temporal split give one realisation of the drift between the development and the Test period; the intervals describe resampling of its days and linked transactions, not other periods. The run has one scoring pass and one completed evaluation attempt. The rolling-origin analysis of the development data ([analysis evidence](../evidence/analysis/README.md)) is a separate supplement.
- **Conditional RQ3 intervals.** The fitted models and the frozen calibrators stay fixed, so the intervals exclude model-development and calibrator-estimation uncertainty ([DR-007](decisions/DR-007-rq3-cost-ratio-and-break-even.md)).
- **Scenario costs.** The cost ratios, the false-alert conversions and the review capacities are declared scenarios: the dataset has no real review costs, capacities or label-report dates ([dataset documentation](dataset.md)); amounts are exposure proxies.
- **Dependence.** The dataset has no account identifier; the inference uses a pseudo-entity key (a heuristic) and day blocks, which bound the dependence that the data allow ([DR-006](decisions/DR-006-bootstrap-inference-strategy.md)); the McNemar tests assume independent frauds and are descriptive.
- **Judged margins.** The SESOI 0.02 and the H3 tolerance of 20 % are declared judgements ([DR-005](decisions/DR-005-sesoi-and-h1-h2-inference.md), [DR-003](decisions/DR-003-rq2-h3-policy-transfer-fidelity.md)); the sensitivity views show how the descriptive positions change with them.
- **Multiplicity.** The Holm procedure covers H1, H2 and H3; every other result, including the CatBoost contrast of H2, the other configurations of H3, RQ3 and the slices, is descriptive or exploratory.
- **Scope.** The models are the eight configurations of DR-011 under one training procedure; no other model family, resampling method or feature engineering was evaluated.

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| δ_F | Family effect: neural minus tree-based Recall@2 %, averaged over the two models of each family and over both weightings |
| Recall@2 % | Share of the Test frauds reviewed at the 2 % review capacity (K = 56 per day) under R3 |
| K | Fixed daily review budget (reviews per relative day) |
| pp, s | Percentage points; seconds |
| p | p-value of a test (bootstrap, Wald or exact, as stated); the Holm-adjusted p-value decides the hypotheses |
| SESOI | Smallest Effect Size Of Interest (0.02 of Recall@2 %); beyond, inside and crossing name the position of an interval against the band ±SESOI |
| H1, H2, H3 | Confirmatory hypotheses of the protocol: family effect, weighting equivalence, alert-rate transfer |
| RQ1, RQ2, RQ3 | Research questions of the protocol: model family and weighting, policy-transfer fidelity, cost-aware capacity use |
| R2, R3, R4 | Decision policies: capacity-rate threshold; daily top-K review; calibrated cost-aware capped policy |
| V2 | Validation 2, relative days 127 to 154 |
| LGB, Cat, RES, FTT, LR | LightGBM, CatBoost, ResNet-style multilayer perceptron (MLP), FT-Transformer (Feature Tokenizer + Transformer), logistic-regression reference; suffix -u unweighted, -b balanced class weighting |
| Holm | Holm step-down procedure over H1, H2 and H3 |
| S1, S2 | Sensitivity analyses: Owen-corrected Wald and day-level moving block |
| t_min | Infimum of the relative tolerances whose open band contains the H3 interval |
| ρ | Miss-to-review cost ratio (scenario assumption) |
| p* | Cost-derived probability threshold 1 / ρ of R4 |
| J, J_R3, J_R4, ΔJ, N_rev, FN | Scenario cost; its value for R3 and for R4; J_R4 − J_R3; number of reviews; number of missed frauds |
| ρ_sat, ρ_BE | Saturation ratio (R4 fills the budget) and break-even ratio (R4 stops being cheaper) |
| Brier, log loss | Mean squared error of the probability; logarithmic loss of the probability |
| intercept, slope | Calibration intercept (target 0) and slope (target 1) of logistic recalibration |
| ROC-AUC | Area under the receiver operating characteristic curve |
| McNemar | Exact test of paired discordant frauds (independence-based, descriptive here) |
| JSON | JavaScript Object Notation |
| DR-NNN | Decision Record NNN |
| REF-NNN | Reference identifier in the reference register |
| [D], [L], [I] | Evidence tags: observed evidence, literature, interpretation |
