# DR-007 — RQ3 cost-ratio grid and break-even rule

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) by Issue #13, pending supervisor review. Resolves protocol open item 4 (cost-ratio grid, role of C_FP, break-even definition). |
| Date | 2026-09-24 |
| Evidence source | Decision-theoretic derivation (checked numerically on synthetic data only); cost-sensitive learning, cost-curve and decision-curve literature; `notebooks/01_dataset_feasibility_audit.ipynb` §§6, 11, 12. **No Test labels, no model scores and no new dataset computations were used.** |
| Related | [DR-002](DR-002-review-capacity-definition.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [DR-005](DR-005-sesoi-and-h1-h2-inference.md), [DR-006](DR-006-bootstrap-inference-strategy.md), [protocol](../protocol.md) |

## Context

Research question 3 (**RQ3**) asks:

> **Under different assumed miss-to-review cost ratios, when does a calibrated cost-aware review policy outperform always filling the available review capacity?**

It compares two decision policies at each review-capacity scenario c (share of transactions that can be reviewed) ∈ {1 %, 2 %, 5 %, 10 %} ([DR-002](DR-002-review-capacity-definition.md)):
- **R3 (daily top-K):** review the K_c highest-scoring transactions on each relative day. K_c is the fixed daily review budget (28 / 56 / 139 / 278).
- **R4 (calibrated cost-aware capped policy):** review a transaction only if its calibrated fraud probability reaches a cost-derived threshold, and never more than K_c per day.

The working protocol left four things open (open item 4):
- the cost-ratio grid (candidate: {1, 2, 5, 10, 20, 50, 100});
- whether a false-alert / customer-friction cost is part of the primary model;
- the exact break-even definition;
- how amount-based results are reported.

**R3 and R4 are research evaluation policies, not the rules of any company or bank. Every cost value in this record is a simulated research assumption.** Nothing here estimates a real banking cost, loss or saving.

## Literature findings

### Cost-derived thresholds from calibrated probabilities

- **[L]** For a two-class decision with cost matrix entries c(predicted, actual), the expected-cost-minimising rule predicts class 1 if and only if P(class 1 | x) ≥ p*, where x is the example's features and [REF-008, Eq. 2]:

  ```text
  p* = (c₁₀ − c₀₀) / (c₁₀ − c₀₀ + c₀₁ − c₁₁)
  ```

  where:

  - `p*` = probability threshold at or above which predicting class 1 has the lower expected cost;
  - `cᵢⱼ` = cost of predicting class i when the actual class is j (i, j ∈ {0, 1}; 1 = positive class);
  - `c₁₀` and `c₀₁` = costs of a false positive and a false negative; `c₀₀` and `c₁₁` = costs of the two correct decisions.

  The numerator is the extra cost of acting on a true negative. The denominator adds the saving from acting on a true positive. p* is the probability at which acting and not acting have equal expected cost.
  - Elkan requires the "reasonableness" conditions c₁₀ > c₀₀ and c₀₁ > c₁₁. When one of them fails, one decision is optimal for every example.
  - Optimal decisions do not change when every cost is multiplied by a positive constant or shifted by a constant, so "any 2x2 cost matrix has essentially only one degree of freedom from a decision-making perspective" [REF-008].
  - Elkan also warns that every cost must be measured from the **same baseline**, otherwise the matrix is economically incoherent. He recommends fitting the classifier on the data as given and then applying the cost threshold to its probability estimates [REF-008].
- **[L]** **Fraud-specific cost matrix.** Höppner et al. use: false positive cost = true positive cost = a fixed investigation cost c_f (the cardholder is contacted in both cases), false negative cost = the transaction amount A_i, true negative cost = 0. The optimal threshold for transaction i is then:

  ```text
  t*_i = c_f / A_i
  ```

  where:

  - `t*_i` = optimal probability threshold for transaction i;
  - `c_f` = fixed investigation cost, the same for every alert;
  - `A_i` = amount of transaction i (the cost of missing it if it is fraud).

  A transaction is worth investigating when its fraud probability exceeds its investigation cost divided by its amount. They stress that this requires "true calibrated probabilities rather than just scores that rank" [REF-073]. They attribute the fixed-cost matrix to Hand et al. [REF-006]. This is exactly the structure of the protocol's primary cost model, with an example-dependent missed-fraud cost.
- **[L]** **Threshold choice and calibration.** Choosing the threshold directly from the probability scale ("score-driven" threshold choice) gives an expected loss, averaged uniformly over cost proportions, equal to the Brier score. If the model is perfectly calibrated, this equals the loss of the optimal threshold choice [REF-038]. A policy that fixes the alert *rate* (such as R3) belongs to the "rate-driven" family, whose expected loss is linked to ranking quality (AUC, the area under the ROC curve) instead [REF-038].

### Uncertain costs, cost curves and decision curves

- **[L]** When costs cannot be specified precisely, classifiers should be compared across the whole range of plausible operating conditions, not at one assumed cost [REF-039].
  - Provost & Fawcett name fraud detection as a domain where "costs change based on workforce and reimbursement issues".
  - They treat the fixed-workforce problem ("the system should present the best possible set of K cases") as a separate decision criterion. This is R3's criterion [REF-039].
- **[L]** **Cost curves** plot normalised expected cost against a probability-cost operating condition. Each fixed classifier is a straight line, and "the intersection point of the two lines is the value … where [the two classifiers] perform identically" [REF-040]. That is the classical break-even point. Drummond & Holte also put bootstrap confidence intervals on cost lines [REF-040].
- **[L]** **Decision curve analysis** (medical prediction) plots the *net benefit* of a model against the threshold probability. The threshold is taken to encode how the decision-maker weighs false positives against false negatives [REF-072]. The ratio is described as an "exchange rate", for example "the maximum number of patients a doctor would recommend for biopsy to find one cancer". Because it is subjective, net benefit is plotted across "a range of reasonable exchange rates" [REF-075]. This is the same logic as RQ3's ratio grid. Here ρ (the miss-to-review cost ratio: assumed cost of missing a fraud ÷ cost of one review) is the largest number of reviews worth spending to find one fraud.
- **[L]** **Calibration drives decision value.** In simulation, "miscalibration always reduced Net Benefit". Harm, meaning a result worse than the default strategies, arose when a model **under-estimated risk at a threshold below the event rate**, or over-estimated risk at a threshold above it [REF-074].

### Break-even analysis in fraud detection

- **[L]** No verified source prescribes a cost-ratio grid or a break-even rule for capacity-constrained fraud review. The fraud literature supplies the cost-matrix structure [REF-006, REF-073] and the capacity framing [REF-004, REF-005, REF-006]. The grid and break-even rule below apply the general methods above and are **researcher decisions [I]**.
- **[L]** The protocol cites REF-007 for amount-dependent costs. Its author copy could not be read in this pass (access restricted), so the amount-dependent cost structure is supported here by REF-073, which was read in full (arXiv version).

## Derivation and validity of the current equations

### Per-transaction decision (no capacity limit)

Assumptions (all simulated, all stated):
- **A1 — Perfect verification.** A reviewed fraud is caught, with no residual loss. A reviewed legitimate transaction is cleared.
- **A2 — Per-transaction, additive costs.** Reviewing one transaction does not stop other frauds of the same entity. Costs do not depend on each other.
- **A3 — Common baseline.** An unreviewed legitimate transaction costs 0. All costs are measured from that baseline [REF-008].
- **A4 — C_R is charged on every review**, fraud or not. C_FP is charged only on reviewed legitimate transactions. C_miss is charged only on unreviewed frauds.
- **A5 — Calibration.** p̂_i equals the Test-period probability that transaction i is fraud.
- **A6 — Unused capacity has value.** C_R is the marginal cost a review slot would incur or free up (e.g. analyst time redeployable to other work). If slots were fully sunk with no alternative use, C_R = 0 and only C_FP could justify leaving capacity unused. The reduction below covers that case as well.

**Direct derivation.** For a transaction with calibrated fraud probability p̂, the expected cost of each action is:

```text
E[cost | not reviewed] = p̂ · C_miss

E[cost | reviewed]     = C_R + (1 − p̂) · C_FP
```

where:

- `p̂` = calibrated probability that the transaction is fraud;
- `C_R` = cost of reviewing one transaction;
- `C_miss` = cost of missing a fraud;
- `C_FP` = additional false-alert/customer-friction cost for reviewing a legitimate transaction;
- `E[cost | action]` = expected cost of taking that action.

Not reviewing costs C_miss only if the transaction is fraud. Reviewing always costs C_R, plus C_FP if the transaction turns out to be legitimate.

Review when reviewing is no more expensive in expectation:

```text
C_R + (1 − p̂) · C_FP  ≤  p̂ · C_miss

C_R + C_FP − p̂ · C_FP  ≤  p̂ · C_miss

C_R + C_FP  ≤  p̂ · (C_miss + C_FP)

p̂  ≥  (C_R + C_FP) / (C_miss + C_FP)
```

The right-hand side of the last line is the threshold p*.

**Check against Elkan's formula.** In Elkan's notation (class 1 = review/fraud), the cost matrix entries are:

```text
c₀₀ = 0             (legitimate, not reviewed)
c₀₁ = C_miss        (fraud, not reviewed)
c₁₀ = C_R + C_FP    (legitimate, reviewed)
c₁₁ = C_R           (fraud, reviewed)
```

where:

- `cᵢⱼ` = cost of action i (1 = review, 0 = do not review) when the true class is j (1 = fraud, 0 = legitimate).

Substituting into REF-008 Eq. 2, step by step:

```text
p* = (c₁₀ − c₀₀) / (c₁₀ − c₀₀ + c₀₁ − c₁₁)

p* = (C_R + C_FP) / ((C_R + C_FP) + C_miss − C_R)

p* = (C_R + C_FP) / (C_miss + C_FP)

p* = 1 / ρ′
```

where:

- `p*` = cost-derived probability threshold used by R4;
- `ρ′` = effective cost ratio, (C_miss + C_FP) / (C_R + C_FP).

The two routes agree. The threshold is the reciprocal of the effective cost ratio ρ′ = (C_miss + C_FP) / (C_R + C_FP).

**Primary case (no false-alert cost).** When C_FP = 0:

```text
p* = C_R / C_miss

p* = 1 / ρ
```

where:

- `ρ` = C_miss / C_R, the miss-to-review cost ratio.

A transaction is reviewed when its calibrated fraud probability is at least one divided by the miss-to-review cost ratio. For example, with ρ = 20 it is reviewed if p̂ ≥ 0.05, i.e. if at least one review in twenty is expected to find a fraud.

**Verdict [I].** Both equations in the protocol are **correct under A1–A5**. They depend critically on A4.
- If C_R were wrongly charged only on false alerts, the threshold would become the following. That is a different and incorrect rule under the stated assumptions:

  ```text
  p*_wrong = C_R / (C_R + C_miss)
  ```

  where:

  - `p*_wrong` = the threshold that would result if C_R were charged only on reviewed legitimate transactions (the incorrect variant).

- Elkan's reasonableness conditions reduce to:

  ```text
  C_miss > C_R,   i.e.   ρ > 1
  ```

  For ρ ≤ 1 the threshold is p* ≥ 1. Reviewing can then never reduce expected cost, even for a transaction that is certainly fraud, so R4 reviews nothing.

### With the daily capacity limit

On each day, R4's rule is equivalent to maximising the total expected net benefit of reviewing over at most K_c transactions. The expected net benefit of reviewing transaction i is:

```text
b_i = p̂_i · (C_miss + C_FP) − (C_R + C_FP)
```

where:

- `b_i` = expected net benefit of reviewing transaction i, in the same units as the costs;
- `p̂_i` = calibrated fraud probability of transaction i;
- `C_miss`, `C_R`, `C_FP` = missed-fraud, review and false-alert costs (as defined above).

b_i is the expected cost avoided by reviewing transaction i, minus the expected cost of the review itself. It is positive exactly when p̂_i ≥ p*.

- b_i increases with p̂_i, so the optimum is: take transactions in descending p̂ order while b_i ≥ 0, up to K_c.
- Platt scaling is monotone, so descending p̂ is the same order as descending raw score. Hence:

```text
R4_d(ρ) = { i ∈ R3_d : p̂_i ≥ p* }
```

where:

- `R4_d(ρ)` = set of transactions that R4 reviews on day d at cost ratio ρ;
- `R3_d` = R3's queue on day d: the K_c transactions with the highest raw scores;
- `K_c` = fixed daily review budget at capacity c;
- `p*` = R4's threshold (1/ρ in the primary model; 1/ρ′ when C_FP is included).

On each day d, R4 reviews those transactions in R3's top-K_c queue whose calibrated probability reaches the threshold. That is the top min(K_c, number with p̂ ≥ p*) transactions of R3's queue.

Consequences:
- **R4 ⊆ R3 on every day.** R4 can only leave capacity unused; it never reviews a transaction that R3 skips.
- **Under A5, R4 minimises expected scenario cost among all policies that review at most K_c per day, and so it cannot be worse than R3 in expectation [I].** An observed loss for R4 against R3 therefore indicates calibration failure at the margin, or chance. This is the central interpretive point of RQ3.

This was checked numerically on synthetic data (brute-force search over all subsets for small cases). No project data were used.

### C_FP reduces to a relabelled ratio

Define the scenario cost of policy π, in units of C_R (all costs divided by C_R):

```text
J_π(ρ, φ) = N_π + φ · FP_π + ρ · FN_π

FP_π = N_π − TP_π
```

where:

- `π` = a decision policy (R3 or R4);
- `J_π(ρ, φ)` = scenario cost of policy π, in units of one review cost;
- `N_π` = number of reviews; `TP_π` = frauds reviewed; `FP_π` = legitimate transactions reviewed; `FN_π` = frauds not reviewed;
- `ρ` = C_miss / C_R; `φ` = C_FP / C_R.

The scenario cost counts one unit per review, φ units per reviewed legitimate transaction, and ρ units per unreviewed fraud.

**Difference between R4 and R3, step by step.** Compared with R3, R4 makes u fewer reviews. Of these, f are frauds and (u − f) are legitimate. R4 therefore also misses f more frauds:

```text
J_R4 − J_R3 = −u − φ · (u − f) + ρ · f

J_R4 − J_R3 = (ρ + φ) · f − (1 + φ) · u

J_R4 − J_R3 = (1 + φ) · (ρ′ · f − u)
```

where:

- `u` = unused reviews: reviews made by R3 but not by R4 (N_R3 − N_R4);
- `f` = frauds forgone: frauds reviewed by R3 but not by R4 (TP_R3 − TP_R4);
- `ρ′` = effective cost ratio (defined next).

The effective cost ratio, written in terms of ρ and φ, is:

```text
ρ′ = (ρ + φ) / (1 + φ)
```

This is the same ρ′ = (C_miss + C_FP) / (C_R + C_FP) defined above, after dividing numerator and denominator by C_R. The factor (1 + φ) is positive, so it never changes which policy is cheaper.

**Consequence [I].** Comparing R4 with R3 depends on the three costs **only through ρ′**, up to a positive scale factor.
- The C_FP model therefore adds no new decision behaviour. It relabels the ratio axis. A break-even ratio ρ_BE found with C_FP = 0 corresponds, under a given φ, to:

  ```text
  ρ = (1 + φ) · ρ_BE − φ
  ```

  where:

  - `ρ_BE` = break-even ratio found in the primary model (C_FP = 0), defined in §4;
  - `ρ` = the miss-to-review cost ratio that gives the same R4-vs-R3 comparison when false alerts also cost φ review units.

  This is ρ′ = ρ_BE solved for ρ. It converts a primary break-even into the missed-fraud cost ratio that gives the same decision when false alerts also carry friction cost φ.
- This holds because C_FP is charged per reviewed legitimate transaction, and every policy compared here reviews a subset of R3's queue. It would not hold for a policy that blocks transactions without review.

## Decision

### 1. Primary cost model: A (review cost and missed-fraud cost only)

```text
J_π(ρ) = N_π + ρ · FN_π

p* = 1 / ρ
```

where:

- `J_π(ρ)` = scenario cost of policy π (R3 or R4), in units of one review cost;
- `N_π` = number of reviews; `FN_π` = frauds not reviewed;
- `ρ` = C_miss / C_R, the miss-to-review cost ratio;
- `p*` = R4's review threshold.

The primary scenario cost, in units of one review cost, is the number of reviews plus ρ times the number of frauds not reviewed. R4's threshold is the reciprocal of ρ.

Why [I]:
- **No loss of generality for RQ3.** R4-vs-R3 comparisons under model B (with C_FP) are the primary results with ρ read as ρ′ (derivation above).
- **One parameter** gives a single cost-difference curve and one break-even ratio per capacity. Model B would need a two-dimensional surface for the same information.
- **No dataset support for C_FP.** IEEE-CIS contains no information on customer contact, declines or friction [D; feasibility notebook §12].
- **Literature fit.** The fixed-investigation-cost fraud matrix [REF-073, attributed to REF-006] is model A.

**C_FP is a sensitivity analysis only.** Break-even ratios are converted for φ ∈ {1, 5} (friction equal to one and five review costs, both illustrative), using the conversion ρ = (1 + φ) · ρ_BE − φ derived above. No separate computation is needed. Relative cost differences are reported for φ = 0 only, because J_R3 itself depends on φ.

### 2. Ratio grid

> **Tabulated grid: ρ ∈ G = {2, 5, 10, 20, 50, 100, 200, 500, 1000}** (1–2–5 logarithmic sequence). Same grid at every capacity and for every configuration.
>
> **Continuous analysis range: ρ ∈ (1, 1000]**, evaluated exactly (see §4).

Assessment of the candidate {1, 2, 5, 10, 20, 50, 100}:

| Question | Finding | Decision |
|---|---|---|
| Ratio 1 | p* = 1: R4 reviews nothing. Elkan's reasonableness boundary [REF-008]. Degenerate | **Removed** from the grid; kept as the open lower end of the continuous range |
| Ratios below 1 | Reviewing never reduces expected cost (p* > 1) | **Not needed** |
| Spacing | p* = 1/ρ, so thresholds are equally spaced on a log scale under log spacing. Cost ratios are naturally multiplicative [REF-039, REF-040] | **Log (1–2–5)** |
| Upper end 100 (p* = 0.01) | Under calibration, the saturation ratio has a lower bound close to capacity ÷ prevalence (0.30 / 0.59 / 1.48 / 2.95; derivation below the table) [I, from D]. This is only a lower bound. At 10 %, K_c = 278 against ≈ 94 frauds per day, so the 278th-ranked transaction can have a p̂ well below 0.01. A grid ending at 100 risks break-evens that are **censored** (still unresolved at the top of the grid) | **Extended to 1000** (p* = 0.001, about 1/34 of V2 prevalence), so that R4 is expected to coincide with R3 at every capacity by the top of the range |
| Different grids per capacity | The ratio describes the cost assumption, not the capacity. A common grid makes capacities comparable | **Same grid** |
| Continuous estimate | Exact evaluation is possible (§4) | **Yes**, in addition to the grid |

**Lower bound on the saturation ratio [I].** Let F̄_d be the expected number of frauds on day d. Under calibration, the day's calibrated probabilities sum to F̄_d, and the K_c-th largest of them cannot exceed the average of the top K_c:

```text
p̂_(K_c),d  ≤  (1 / K_c) · Σ_{top K_c} p̂_i  ≤  F̄_d / K_c

ρ_sat  =  1 / min_d p̂_(K_c),d  ≳  K_c / F̄_d  ≈  c / prev_V2
```

where:

- `p̂_(K_c),d` = calibrated probability of the K_c-th ranked transaction on day d;
- `Σ_{top K_c} p̂_i` = sum of the calibrated probabilities of the day's K_c highest-ranked transactions;
- `F̄_d` = expected number of frauds on day d;
- `ρ_sat` = saturation ratio (defined in §4);
- `c` = capacity share; `K_c` = daily review budget; `prev_V2` = fraud prevalence in Validation 2 (V2, relative days 127–154), 3.39 %.

R4 cannot fill a budget larger than the expected fraud count unless the ratio is at least about capacity ÷ prevalence. Realistic saturation ratios are expected to be much higher than this bound.

**Reference ratio.** The ratio at which p* equals V2 fraud prevalence prev_V2 = 3.39 % [D] is marked on every figure:

```text
ρ_ref = 1 / prev_V2 ≈ 29.5
```

At this ratio the review threshold equals the base rate. Below it, reviewing an average transaction costs more than it saves.

High ratios are included to show where the policies merge, **not because they are claimed to be realistic**. No grid value is a real cost ratio.

### 3. Evaluation of R4 across the capacity grid

For each capacity c, configuration m and ratio ρ:
- **R4 queue** on day d:

  ```text
  R4_d(ρ) = { i ∈ R3_d : p̂_i ≥ 1 / ρ }
  ```

  where:

  - `R4_d(ρ)` = R4's reviewed set on day d at ratio ρ;
  - `R3_d` = R3's top-K_c queue on day d;
  - `p̂_i` = calibrated fraud probability of transaction i.

  R4 keeps those transactions in R3's daily queue whose calibrated probability reaches 1/ρ.
  - **Ranking is always by raw score s_i**, using R3's pre-declared tie rule ([DR-006](DR-006-bootstrap-inference-strategy.md)). p̂_i is used only for the threshold comparison.
  - With isotonic calibration (not strictly monotone), this still gives a prefix of R3's queue.
- **Threshold ties:** review if p̂_i **≥** p*. At exact indifference R4 keeps R3's action (fill capacity), so R4 departs from R3 only where it expects a strict gain [REF-008 treats equality as indifference].
- **No tuning.** p* is fixed by ρ. Nothing is selected on V2 or Test. All ratios are reported.
- **All four configurations are reported.** RQ3 selects no "best" configuration.

### 4. Break-even definitions

Over the 28 Test days, the **unused reviews** are:

```text
u(ρ) = Σ_d ( K_c − |R4_d(ρ)| )
```

where:

- `Σ_d` = sum over the 28 Test days;
- `K_c` = daily review budget (the size of R3's queue each day);
- `|R4_d(ρ)|` = number of transactions R4 reviews on day d.

This is the number of review slots that R4 leaves empty, summed over days. It needs no labels.

The **frauds forgone** are:

```text
f(ρ) = Σ_d  Σ_{i ∈ R3_d \ R4_d(ρ)}  y_i
```

where:

- `R3_d \ R4_d(ρ)` = transactions in R3's queue on day d that R4 does not review;
- `y_i` = fraud label of transaction i (1 = fraud, 0 = legitimate).

This is the number of frauds among the transactions that R3 reviews but R4 does not.

The **scenario-cost difference**, step by step (primary model, φ = 0):

```text
ΔJ(ρ) = J_R4(ρ) − J_R3(ρ)

ΔJ(ρ) = (N_R4 − N_R3) + ρ · (FN_R4 − FN_R3)

ΔJ(ρ) = −u(ρ) + ρ · f(ρ)

ΔJ(ρ) = ρ · f(ρ) − u(ρ)
```

where:

- `J_R4(ρ)`, `J_R3(ρ)` = primary scenario costs of R4 and R3 (reviews + ρ × missed frauds);
- `N_R4 − N_R3` = −u(ρ), because R4 makes u fewer reviews;
- `FN_R4 − FN_R3` = f(ρ), because R4 misses the f frauds among those unused reviews.

R4 saves u reviews and pays ρ for each of the f extra frauds it misses. A negative ΔJ means R4 is cheaper.

The literal question, *at what ratio do R4 and R3 have equal cost?*, is degenerate. R4's threshold moves with ρ, so above some ratio R4 fills capacity every day and ΔJ = 0 by identity. Two quantities are therefore defined.

**(a) Saturation ratio (label-free):**

```text
ρ_sat,c,m = 1 / min_d p̂_(K_c),d
```

where:

- `c` = capacity scenario; `m` = model configuration;
- `min_d` = minimum over the 28 Test days;
- `p̂_(K_c),d` = calibrated probability of the K_c-th ranked transaction on day d.

The saturation ratio is set by the day whose last budgeted transaction has the lowest calibrated probability. Once ρ is large enough to admit that transaction, R4 fills the budget on every day.

- For ρ ≥ ρ_sat, R4 ≡ R3.
- For ρ < ρ_sat, R4 leaves capacity unused on at least one day.
- This answers *below which ratio would a calibrated cost-aware rule leave capacity unused at all?*
- It is also computed on V2 (in-sample for the calibrator) as a **pre-Test expectation**.

**(b) Break-even ratio (label-based) — the headline RQ3 output:**

```text
ρ_BE,c,m = sup { ρ ∈ (1, 1000] : ΔJ_c,m(ρ) < 0 }
```

where:

- `sup` = supremum: the largest value (least upper bound) of the set;
- `(1, 1000]` = continuous analysis range of ρ;
- `ΔJ_c,m(ρ)` = scenario-cost difference J_R4 − J_R3 for capacity c and configuration m (negative = R4 cheaper).

It is the **largest assumed ratio at which the cost-aware policy is strictly cheaper than filling capacity**. At every ratio above it, filling capacity is at least as cheap. By construction ρ_BE ≤ ρ_sat.

**Exact computation, no interpolation.**
- R4's decisions change only at the finitely many ratios ρ = 1/p̂_i.
- Between two consecutive change points, u and f are constant and ΔJ is linear in ρ. Any zero crossing inside a segment is therefore exact:

  ```text
  ρ · f − u = 0

  ρ = u / f
  ```

  Within a segment, the policies cost the same at the ratio equal to unused reviews divided by frauds forgone.
- ΔJ is therefore computed exactly on the whole range. **Interpolation between grid points is not used.**
- Computing ΔJ at any ρ from the frozen Test probabilities selects nothing, so it is not tuning.

**Classification of each (c, m) result:**

| Case | Condition | Reported as |
|---|---|---|
| Coincidence break-even | ρ_BE = ρ_sat | R4 is cheaper right up to the ratio at which it merges with R3. The expected pattern under good calibration |
| Crossing break-even | ρ_BE < ρ_sat | R4 becomes dearer than R3 before the policies merge. Leaving capacity unused was costly on (ρ_BE, ρ_sat). Indicates calibration shortfall at the margin |
| No break-even in range | ΔJ ≥ 0 for all ρ ∈ (1, 1000] | "R4 never cheaper than R3 within the declared range" |
| Right-censored | ρ_sat > 1000 and ΔJ(1000) < 0 | "Break-even above 1000" |

- **Non-monotone curves:** if {ρ : ΔJ < 0} is not a single interval, every sign change is reported, and the result is flagged for interpretation from the full curve.
- **Equal cost:** R4 must be *strictly* cheaper to "outperform". ΔJ = 0 below ρ_sat is reported as a tie.

**Fixed-policy break-even (supporting).** Fix the R4 queue built at a grid ratio ρ₀, and ask at which true ratio that fixed queue and R3 would cost the same:

```text
ρ_fixed(ρ₀) = u(ρ₀) / f(ρ₀)

ρ_fixed(ρ₀) = 1 / ( f(ρ₀) / u(ρ₀) )
```

where:

- `ρ₀` = the grid ratio used to build R4's queue (and its threshold p*(ρ₀) = 1/ρ₀);
- `ρ_fixed(ρ₀)` = true ratio at which that fixed R4 queue and R3 have equal scenario cost;
- `f(ρ₀) / u(ρ₀)` = fraud rate among the reviews R4 left unused.

This break-even is the reciprocal of the fraud rate among the unused reviews. It is the classical crossing point of two cost lines [REF-040].

R4 at ρ₀ beats R3 at its own ratio exactly when:

```text
ρ₀ · f(ρ₀) − u(ρ₀) < 0

f(ρ₀) / u(ρ₀) < 1 / ρ₀

f(ρ₀) / u(ρ₀) < p*(ρ₀)
```

In words, R4 wins if the reviews it left unused contained frauds at a lower rate than the threshold it assumed.

**Scope of the break-even.** ρ_sat and ρ_BE are computed **separately for each capacity and each configuration**, because both depend on the model's calibrated probabilities.

**Robustness caveat.** ρ_BE is determined by the few lowest-ranked transactions in R3's queue near ρ_sat, so it can move with a handful of transactions. It is **never reported alone**: it always appears with the curve and the grid-point intervals (§7).

### 5. Capacity interaction [I]

- **What "binding" means here.** "Binding" (1 %, 2 %) in [DR-002](DR-002-review-capacity-definition.md) is an *oracle* property: K_c is below the daily fraud count. What matters for R4 is the *model's* calibrated probability at rank K_c. That value is below 1 and unknown until models exist.
  - At 1 % and 2 %, the rank-K transaction is still expected to have a relatively high p̂. So ρ_sat is expected to be **low** (plausibly single digits), and R4 ≡ R3 for most of the grid.
  - **The DR-002 expectation holds, with a qualification.** At 2 %, R4 is expected to equal R3 above a low ρ_sat. RQ3 at 2 % is informative only at low ratios, and ρ_sat,2% is itself a finding.
- **Slack capacities (5 %, 10 %).** K_c exceeds daily frauds on every V2 day [D], so R3 must fill its queue with low-probability transactions.
  - The lower bound ρ_sat ≳ capacity ÷ prevalence (1.48, 2.95) is loose. Realistic values are expected to be one or two orders of magnitude higher.
  - These capacities are where R4 leaves capacity unused across a **wide** range of ratios, and they remain the most informative for RQ3.
- **R4 does not depend on capacity at low ratios.** Below a given ratio, R4 wants fewer reviews than any K_c, so R4 is the same queue at every capacity. Cross-capacity differences in ΔJ then come from R3 alone.
- **Calibration risk depends on capacity** [REF-074].
  - At 5 % and 10 %, the relevant thresholds lie **below the event rate** (ρ > 29.5). There, under-estimated risk (e.g. Test prevalence above V2's) is the harmful direction.
  - At 1 % and 2 %, the relevant thresholds lie above the event rate. There, over-estimation is the harmful direction.

### 6. Amount-based analysis

- **The amount analysis is an evaluation-only sensitivity analysis.** R3 and R4 decisions stay count-based (the threshold uses ρ). No amount-aware policy is added: a transaction-specific threshold C_R / A_i [REF-073] would be a **new decision policy** and would need its own GitHub Issue.
- **Reported quantities [D]:**
  - `TransactionAmt` (the dataset's transaction amount) is in United States dollars (USD) per the organiser [REF-048].
  - Amount-weighted recall of R3 and R4.
  - **Fraud value exposed**: Σ amount of unreviewed frauds, for each policy.
  - **Fraud value forgone** by R4: Σ amount of the frauds among the unused reviews.
  - **Fraud value forgone per unused review** (USD per unused review).
- **Naming rule.** These are **exposure proxies**, never "loss", "saving" or "investigation cost". They are not converted into a break-even review cost in USD, because that would require assuming that a missed fraud costs exactly its transaction amount. That is an unverified loss model: no chargeback, recovery, fee or investigation data exist [D; feasibility notebook §12].
- **Uncertainty.** Fraud value is highly concentrated: the top 20 % of development frauds carry 62 % of fraud value [D]. Amount-based intervals will be wide, and single large frauds can dominate them.

### 7. Statistical treatment

**RQ3 remains estimation and sensitivity analysis. No confirmatory hypothesis is added. The confirmatory hypotheses H1 (model family), H2 (class weighting) and H3 (alert-rate transfer), the SESOI (smallest effect size of interest), the Holm family and the H1/H2 bootstrap ([DR-006](DR-006-bootstrap-inference-strategy.md)) are unchanged.**

Why [I]:
- ρ is an assumption, not a population parameter. A test at each ratio would test a family of assumption-conditional nulls.
- ρ_sat is a minimum over days, and ρ_BE the supremum of a step function. Standard bootstrap intervals are unreliable for such non-smooth statistics, so a confirmatory test on them would rest on weak inference.
- Adding RQ3 to the Holm family would reduce power for H1–H3.

Uncertainty reporting (descriptive):
- **Grid-point intervals.** For each (c, m, ρ ∈ G): 95 % percentile confidence intervals (CIs) for ΔJ, relative ΔJ, u, f and the amount measures.
  - Method: a **day-level moving-block bootstrap (block length ℓ = 3 days, B = 10,000)**, the same key-free machinery as H3 and sensitivity analysis S2 of DR-006 (the key-free day-block bootstrap).
  - Whole days are resampled, so each day's R3 and R4 queues stay exactly intact. R3 and R4 are paired within each replicate.
  - **Limitation:** this does not capture cross-day entity dependence of frauds, so intervals for f may be too narrow ([DR-006](DR-006-bootstrap-inference-strategy.md) evidence). As a sensitivity check, the interval for frauds forgone f (a recall difference between R3 and R4) is also computed with the DR-006 two-way pigeonhole bootstrap.
- **Regime map.** Each (c, m, ρ ∈ G) cell is labelled:
  - *identical* (u = 0);
  - *R4 cheaper* (95 % CI of ΔJ below 0);
  - *R3 cheaper* (CI above 0);
  - *inconclusive*.

  No multiplicity adjustment is applied. Patterns across adjacent ratios are interpreted, not isolated cells.
- **ρ_sat and ρ_BE** are reported as point estimates without CIs, for the reason above. Their uncertainty is conveyed by the grid-point intervals.

### 8. Calibration

**No change to the calibration plan.** No methodological conflict was found.
- **Why calibration matters for R4.** R4 compares p̂ with an absolute threshold. R3 and every rate-based policy use only the ranking [REF-038, REF-073]. Miscalibration always lowers decision value and can make the cost-aware rule worse than its default alternative [REF-074].
- **Primary: Platt scaling fitted on V2**, as in the protocol.
- **Secondary check: isotonic regression.** RQ3 is recomputed with isotonic p̂. Any change in the ρ_BE classification or the regime map is reported as calibration sensitivity. Confirmatory interpretation uses Platt only.
- **Declined-set calibration diagnostic.** For each grid ratio, the realised fraud rate among unused reviews, f/u, is reported next to (i) the mean p̂ of those transactions and (ii) p*. This is the calibration that decides RQ3.
- **Interpretation rule when calibration is poor.** If Test calibration metrics (intercept, slope, reliability diagram) or the declined-set diagnostic show material miscalibration, R4 results are reported as the joint effect of **the cost rule and calibration transfer**. They are **not** evidence for or against cost-aware reviewing as such. A crossing break-even (ρ_BE < ρ_sat) is read as a calibration-transfer finding.
- **Label-latency supplement.** The protocol's latency sensitivity (calibrator refitted on V2 days 127–147) applies to R4 and is reported for RQ3 as exploratory.

### 9. RQ3 outputs

Reported for every capacity c, configuration m and grid ratio ρ ∈ G, plus exact curves over (1, 1000]:

| Output | Definition | Source type |
|---|---|---|
| R3 reviews used | N_R3 = 28 · K_c | Dataset-supported, label-free [D] |
| R4 reviews used; unused reviews u | N_R4(ρ); u = N_R3 − N_R4 | Dataset-supported, label-free [D] |
| R4 utilisation | N_R4 / (28 · K_c); days with unused capacity | Dataset-supported, label-free [D] |
| Saturation ratio ρ_sat (Test; V2 pre-Test expectation) | §4(a) | Dataset-supported, label-free [D] |
| Frauds captured and false negatives, R3 and R4 | TP_π, FN_π; recall; normalised recall | Dataset-supported, label-based [D] |
| Frauds forgone f; fraud rate among unused reviews f/u | §4 | Dataset-supported, label-based [D] |
| Declined-set calibration | f/u vs mean p̂ vs p* | Dataset-supported [D] |
| Amount-weighted recall; fraud value exposed; fraud value forgone; forgone value per unused review | §6 (USD, exposure proxy) | Dataset-supported proxy [D] |
| Scenario cost J_R3, J_R4 | Review-cost units, total and per Test day; no-review reference J = ρ · F, where F = number of Test frauds | **Scenario-derived** |
| ΔJ and relative ΔJ = ΔJ / J_R3 | With 95 % CIs at grid points | **Scenario-derived** |
| Break-even ratio ρ_BE and its classification | §4(b) | **Scenario-derived** |
| Fixed-policy break-even ρ_fixed = u/f | §4 | **Scenario-derived** |
| Regime map | §7 | **Scenario-derived** |
| C_FP-converted break-evens (φ = 1, 5) | ρ = (1 + φ) ρ_BE − φ | **Scenario-derived** |
| ρ grid, φ values, C_R = 1 unit, assumptions A1–A6 | — | **Purely assumed parameters** |

**Figures:**
- **Cost-difference curve:** relative ΔJ against log ρ, one panel per capacity, one line per configuration, with ρ_sat, ρ_BE and 1/prev_V2 marked.
- **Utilisation curve:** R4 utilisation against log ρ.

The cost-difference plot follows the cost-curve idea of plotting performance across operating conditions [REF-040, REF-072]. It is not a formal Drummond–Holte cost curve, which assumes uncapacitated classifiers and normalises costs differently.

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Keep the candidate grid {1, …, 100} | Ratio 1 is degenerate. An upper end of 100 risks censored break-evens at 5 % and 10 % |
| Linear grid | Thresholds p* = 1/ρ would be crowded at small p* and sparse at large p*. Cost ratios are multiplicative |
| Capacity-specific grids | Would hide cross-capacity comparison; the ratio is a cost assumption, not a capacity property |
| Model B (with C_FP) as primary | Adds a second free parameter with no data support, and no new decision behaviour (reduces to ρ′) |
| Break-even as ΔJ(ρ) = 0 literally | Degenerate: true for every ρ ≥ ρ_sat by identity |
| Linear interpolation between grid points | Unnecessary: the exact curve is available. Interpolation would add approximation error |
| Amount-aware R4 (threshold C_R / A_i) | A new decision policy; outside the current scope without a new Issue |
| Confirmatory test for RQ3 | Assumption-conditional nulls; non-smooth break-even statistics; would dilute Holm power for H1–H3 |
| Pigeonhole bootstrap as RQ3 primary | DR-006 defines it for fraud-level capture indicators. ΔJ also involves legitimate-transaction counts, for which no entity weighting has been specified. It is used as a sensitivity check for f only |

## Consequences

- Protocol open item 4 is resolved, pending supervisor review.
- R4's threshold is written as p* = 1/ρ, with review at p̂ ≥ p* and ranking by raw score.
- The Test evaluation must save, for every configuration and Test transaction: raw score, Platt p̂, isotonic p̂ and relative day, plus labels and amounts once the Test is unsealed. These are needed to compute exact ΔJ curves.
- ρ_sat is computed on V2 **before** the Test evaluation, as a pre-registered expectation of where R4 and R3 will merge.
- H1, H2, H3, the SESOI, the H3 equivalence tolerance Δ, the capacity grid, the calibration plan and DR-006 are unchanged.

## Threats and limitations

- **All costs are assumptions.** RQ3 results are conditional on A1–A6 and must never be read as institutional costs, losses or savings.
- **A1 (perfect verification) and A2 (transaction-level costs) are strong.**
  - Real reviews can miss fraud.
  - Stopping one transaction of a compromised entity may prevent later ones, which transaction-level costs ignore ([DR-002](DR-002-review-capacity-definition.md), construct validity).
  - A single constant C_miss ignores the variation in fraud value.
- **A6 (unused capacity has value)** is a modelling choice. If review slots have no alternative use, filling capacity is optimal whenever C_FP = 0.
- **R4's advantage depends on calibration transfer** from V2 to Test. Prevalence and composition shift between periods [D], so calibration may degrade.
- **ρ_BE is sensitive to a few marginal transactions** and has no CI. The grid-point intervals are the main uncertainty statement.
- **The day-block intervals may be too narrow** for fraud counts (cross-day entity dependence, [DR-006](DR-006-bootstrap-inference-strategy.md)). The pigeonhole check on f partly addresses this.
- **Labels are chargeback-derived** and miss unreported fraud [REF-048], so f and FN (false negatives) undercount true fraud.
- **Literature read in part.**
  - REF-072, REF-074 and REF-075 were read at abstract level.
  - REF-073 was read in full in its arXiv version.
  - REF-038's threshold-choice results were read in the arXiv preprint (1112.2640v2), whose abstract matches the published JMLR abstract.
  - REF-039 was read in its arXiv version (cs/0009007).
  - REF-007's cost matrix was not read.

## References

REF-004, REF-005, REF-006, REF-007, REF-008, REF-038, REF-039, REF-040, REF-048\*, REF-072, REF-073, REF-074, REF-075 — see [`../references.md`](../references.md).

\* Partially verified.

## Symbols and notation

Complete reference for every symbol used in this record. Each symbol is also defined where it first appears.

| Symbol | Meaning |
|---|---|
| c | Review-capacity scenario (share of transactions), c ∈ {1 %, 2 %, 5 %, 10 %} |
| K_c | Fixed daily review budget for scenario c, in reviews per relative day ([DR-002](DR-002-review-capacity-definition.md)) |
| d | A Test relative day, d ∈ {155, …, 182} (28 days) |
| m | A model configuration: LightGBM or CatBoost, each unweighted or with balanced class weighting (four configurations) |
| s_i | Raw model score of transaction i (used for ranking) |
| p̂_i | Calibrated fraud probability of transaction i: the Platt-scaled score, with the calibrator fitted on Validation 2 (V2, relative days 127–154) |
| y_i | Fraud label of transaction i (1 = fraud, 0 = legitimate) |
| F | Number of frauds in the evaluation period |
| C_R | **Review cost**: the cost of manually reviewing one transaction, charged on every review whatever the outcome. Interpreted as the *marginal, avoidable* cost of a review slot (see Assumption A6) |
| C_miss | **Missed-fraud cost**: the cost of a fraud that is not reviewed, relative to the same fraud being reviewed and caught |
| C_FP | **False-alert (customer-friction) cost**: an extra cost of reviewing a legitimate transaction, on top of C_R |
| ρ | **Miss-to-review cost ratio**, ρ = C_miss / C_R. The single scenario parameter of RQ3 |
| φ | False-alert-to-review cost ratio, φ = C_FP / C_R (sensitivity analysis only) |
| ρ′ | **Effective cost ratio**, ρ′ = (C_miss + C_FP) / (C_R + C_FP). Equals ρ when C_FP = 0 |
| p* | Cost-derived probability threshold used by R4 |
| N_π, TP_π, FN_π (FN), FP_π | For policy π over the evaluation period: number of reviews; frauds reviewed (true positives); frauds not reviewed (false negatives, F − TP_π); legitimate transactions reviewed (false positives, N_π − TP_π) |
| J_π(ρ) | **Scenario cost** of policy π, in units of one review cost (defined in the text) |
| u(ρ) | **Unused reviews**: reviews made by R3 but not by R4, N_R3 − N_R4 |
| f(ρ) | **Frauds forgone**: frauds reviewed by R3 but not by R4, TP_R3 − TP_R4 |
| ΔJ(ρ) | Scenario-cost difference J_R4 − J_R3 (negative = R4 cheaper) |
| ρ_sat | **Saturation ratio**: the smallest ratio at which R4 uses all K_c reviews on every day (defined in the text) |
| ρ_BE | **Break-even ratio** (defined in the text) |
| R3_d, R4_d(ρ) | Sets of transactions reviewed on day d by R3 (the K_c highest raw scores) and by R4 at ratio ρ |
| p̂_(K_c),d | Calibrated probability of the K_c-th ranked transaction on day d |
| b_i | Expected net benefit of reviewing transaction i |
| G | Tabulated ratio grid {2, 5, 10, 20, 50, 100, 200, 500, 1000} |
| ρ₀ | A grid ratio at which an R4 queue is built (used for the fixed-policy break-even) |
| prev_V2 | Fraud prevalence in V2 (3.39 % of V2 transactions are fraud) |
| cᵢⱼ | Elkan's cost-matrix entry: cost of predicting (acting as) class i when the actual class is j [REF-008] |
| c_f, A_i, t*_i | In Höppner et al. [REF-073]: fixed investigation cost per alert; amount of transaction i; optimal threshold for transaction i |
| p*_wrong | Threshold that would result if C_R were charged only on reviewed legitimate transactions (shown only as the incorrect variant) |
| ρ_fixed(ρ₀) | Fixed-policy break-even: the true ratio at which the R4 queue built at ratio ρ₀ and R3 cost the same (defined in the text) |
| ρ_ref | Reference ratio 1 / prev_V2, at which p* equals V2 fraud prevalence prev_V2 |
| F̄_d | Expected number of frauds on day d (used only in the saturation-ratio bound) |
| [D], [L], [I] | Evidence tags: dataset observation, literature, inference |
| A1–A6 | The six modelling assumptions of the derivation (perfect verification; additive per-transaction costs; common baseline; C_R on every review; calibration; unused capacity has value) |
| AUC, ROC | Area Under the Curve; Receiver Operating Characteristic |
| B | Number of bootstrap replicates (10,000) |
| CI | Confidence interval |
| Δ | H3 equivalence tolerance on the alert rate (defined in DR-003; unchanged here) |
| E[cost \| action] | Expected cost of an action (review or not review) |
| H1, H2, H3 | Confirmatory hypotheses of the protocol: model family; class weighting; alert-rate transfer |
| ℓ | Block length of the day-level moving-block bootstrap (3 days) |
| log | Logarithm (log scale for ρ) |
| R3, R4 | Decision policies: daily top-K review; calibrated cost-aware capped review |
| RQ3 | Research question 3 (cost-aware capacity use) |
| S2 | Sensitivity analysis S2 of DR-006: key-free day-level moving-block bootstrap |
| SESOI | Smallest Effect Size Of Interest |
| sup | Supremum: the largest value (least upper bound) of a set |
| USD | United States dollars |
| x | Feature vector of a transaction, in P(class 1 \| x) |
| Σ_d | Sum over the 28 Test days |
