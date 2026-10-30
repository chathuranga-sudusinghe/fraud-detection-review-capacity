# DR-001 — Temporal split strategy

| Field | Value |
|---|---|
| Status | Accepted (pre-freeze) in line with the University of Greenwich criteria ([criteria checklist](../greenwich-criteria-checklist.md)); supervisor feedback welcome, not a precondition |
| Date | 2026-09-23; revised 2026-10-09 (Issue #54), 2026-10-10 (Issue #62: link to the machine-readable declaration) Issue #159 (rolling-origin supplement design) and 2026-10-30 (Issue #185: authorised Test evaluation) |
| Evidence source | `notebooks/01_dataset_feasibility_audit.ipynb`, §§3–4, 5.1, 14, 15 |
| Related | [DR-002](DR-002-review-capacity-definition.md), [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md), [protocol](../protocol.md) |

## Context

The study needs a leakage-safe temporal evaluation. Three things must be separated in time:
- model fitting;
- model selection (tuning, early stopping);
- decision-policy selection (score thresholds, probability calibration, daily review budget).

Performance is then measured once, on a later untouched period. `TransactionDT` is a time offset in seconds from an **undisclosed** reference point [REF-047]. Only *relative* days can be used, and no calendar interpretation is possible.

The split had to be chosen **before** any target-driven analysis, so that the final Test period stays uncontaminated. It was therefore chosen from transaction **volume only**. The only full-data label check was that `isFraud` exists, is binary and has no missing values.

## Decision

Use a chronological, day-aligned split. Each transaction is assigned a relative day:

$$
\text{relative day} = \left\lfloor \frac{\texttt{TransactionDT}}{86{,}400} \right\rfloor
$$

where:

- `TransactionDT` = time offset in seconds from the undisclosed reference point;
- `86,400` = number of seconds in one day;
- `⌊ · ⌋` = rounding down to a whole number (integer division, `//` in Python).

The partitions are:

| Partition | Relative days | Days | Rows | % rows | Role |
|---|---|---|---|---|---|
| **Train** | 1–105 | 105 | 368,632 | 62.4 | Model fitting. **Models are not refit** on later partitions in the primary design. |
| **V1** (Validation 1) | 106–126 | 21 | 65,544 | 11.1 | Hyperparameter selection and early stopping **only** |
| **V2** (Validation 2) | 127–154 | 28 | 77,822 | 13.2 | Probability calibration (for policy R4, the calibrated cost-aware capped policy); threshold selection (R1, the threshold that maximises F1, the harmonic mean of precision and recall; R2, the capacity-rate threshold); fixed daily review budget K ([DR-002](DR-002-review-capacity-definition.md)); selection of the H3 configuration ([DR-010](DR-010-h3-confirmatory-configuration.md)) |
| **Test** | 155–182 | 28 | 78,542 | 13.3 | **Sealed.** One confirmatory evaluation after the protocol freeze |
| **Total (full joined dataset)** | 1–182 | 182 | 590,540 | 100.0 | Train + V1 + V2 + Test; mutually exclusive and exhaustive |

- **No temporal gap in the primary design.**
- **Rolling-origin supplement for RQ2 (exploratory).** An exploratory, descriptive look at how variable the transfer of an R2 threshold is across several origins inside days 1–154, on the primary configuration only [REF-024, REF-025, REF-046]. The primary configuration is the H3 configuration's model and weighting ([DR-010](DR-010-h3-confirmatory-configuration.md)); the supplement runs after the H3 selection and never changes it, any threshold or the confirmatory plan. It is a **retrospective sensitivity analysis conditional on the final development-selected specification**: that configuration and its hyperparameters were selected on V1 and V2, which lie after the early origins, so it is not a prospective deployment simulation [I]. The design is declared machine-readably in [`config/dataset.yaml`](../../config/dataset.yaml) (`rolling_origin`):
  - **Origins.** The origin day d_o (the first day of the threshold window) is 71, 85 or 99, 14 days apart. At d_o, the primary configuration is refitted on relative days 1 to d_o − 22 (days 1–49, 1–63 and 1–77), leaving a gap of 21 days, as between the end of Train and V2 in the main design; the threshold window is days d_o to d_o + 27 and the transfer window the next 28 days, d_o + 28 to d_o + 55 (the last origin's transfer window is days 127–154).
  - **Refit.** The hyperparameters and the fixed length (the retained rounds or epochs) of the selected trial, the main seed, no search and no early stopping; the training procedure, devices and numerical settings are those of [DR-015](DR-015-common-training-rules.md) and [DR-013](DR-013-training-device-policy.md). Everything learned from data is fitted on the origin's training days only: the balanced class weights and, for a neural model, the shared preprocessing of [DR-012](DR-012-neural-network-input-representation.md) with its declared seed; the main-run artifact is not reused. The batch plan of the selected trial is reused, and a plan that would leave a last micro-batch below the architecture's minimum batch size on an origin's training rows (two rows for the ResNet-style MLP) stops the supplement before any fit. A fixed length chosen on the 105 training days is applied to shorter training sets, and the origins differ in calendar position, training history and refit; the results combine these.
  - **Threshold and budget.** R2 at 2 % capacity on the scores of the threshold window (no labels): the threshold is the score ranked ⌈0.02 N_window⌉ among the N_window transactions of that window, and a transaction is flagged when its score is at least the threshold. The origin's fixed daily budget is K = round(0.02 × the mean daily volume of the threshold window), rounded half up ([DR-002](DR-002-review-capacity-definition.md) §3), and is never recomputed from the transfer window.
  - **Reported.** For each origin: the training days and rows, the threshold, K, the realised rate of the threshold window, the pooled transfer rate r (alerts over transactions of the transfer window), r / 0.02 − 1, the days above and below K, the total overflow and unused capacity, and the rate and volume components of [DR-003](DR-003-rq2-h3-policy-transfer-fidelity.md) (with the window's mean daily volume). Across origins: the origins in chronological order and the number of origins whose r lies strictly inside the H3 band (0.016, 0.024), written "x of 3". No interval or test is given and no claim of equivalence or reliability is made: the origins share training days and their transfer windows overlap.
- **The Test period remains sealed.** Its labels have not been used for any analysis, and must not be until the protocol is frozen and tagged in Git. A Boolean flag alone never opens them; access follows the procedure below.
- **Authorised Test evaluation (Issue #185).** The single confirmatory evaluation runs under this procedure, fixed before any confirmatory code:
  - **Authorisation.** The command takes the name of the protocol freeze tag. It runs only from a clean working tree whose HEAD is the commit of that annotated tag, and records the tag, the commit and the source fingerprint. The training identity of the development run is recorded separately; it may differ from the frozen evaluator's source fingerprint.
  - **Inputs.** Only frozen artefacts: the closed development run (main-seed models and the neural preprocessing artifact), the V2 thresholds and calibrators of the closed analysis run, the H3 selection record and the dataset manifest, each verified by hash before any Test row is read.
  - **Two phases.** Phase 1 scores the Test features of the eight configurations and the logistic-regression reference once, without labels and with the timing instrumentation; the scores are written and hashed. Phase 2 reads the Test labels, verifies the Test `transaction_id_sha256` of the manifest, and computes every outcome from the saved scores.
  - **Consumption and repeats.** A lock prevents concurrent runs. Before the labels are read, a failed phase 1 may be repeated; the failure and its log are kept and reported. Once the labels are read the evaluation is consumed: a failure in phase 2 resumes from the saved, hashed scores (computation only, no rescoring), and the command refuses any new phase 1 for the same tag.
  - **Record.** The run record names the tag, both identities, the phase boundaries, every repeat of phase 1 and every resume of phase 2.

Reproducibility anchors, from the notebook (§4.3):

| Partition | `TransactionDT` range (s = seconds) | SHA-256 (Secure Hash Algorithm, 256-bit) of sorted `TransactionID`s (first 16 hexadecimal characters) |
|---|---|---|
| Train | 86,400 – 9,158,384 | `f3c088aa2f952738` |
| V1 | 9,158,441 – 10,972,793 | `cb1325d41dae02a3` |
| V2 | 10,972,801 – 13,391,998 | `54edb97bc6b82d4b` |
| Test | 13,392,056 – 15,811,131 | `9f266b2bdf2da9f1` |
| All partitions | 86,400 – 15,811,131 | — (per-partition hashes above) |

## Evidence

**Temporal structure [D].**
- 590,540 transactions span relative days 1–182, with every day present.
- Rows are sorted by `TransactionDT`, and `TransactionID` is monotone in time.
- Mean volume is 3,245 per day, with a coefficient of variation (standard deviation ÷ mean) of 0.26.
- Days 1–28 average about 4,500 per day; later 4-week blocks average about 2,700–3,300 per day.
- The lag-7 autocorrelation of daily volume (0.50) is weaker than lag-1 (0.75), so the weekly cycle is weak.

**Partition sizes [D].**
- V2 and Test have near-identical mean volume: 2,779 vs 2,805 rows per day.
- Development fraud counts: Train 12,484 (3.39 %), V1 2,768 (4.22 %), V2 2,637 (3.39 %).
- Test fraud count: not computed.

**Drift between partitions [D].**
- The V1 → V2 base-rate shift is −0.23 on the log-odds scale.
- Identity coverage falls from 27.1 % (Train) to 20.7 % (V1) and 17.5 % (V2).
- Most feature-level shift, measured by the Population Stability Index (PSI), occurs between Train and the validation periods, not between V1 and V2.

**Entity-level label dependence [D].** About 16.9 % of V2 frauds come from heuristic pseudo-entities that already had a Train fraud label. Transactions from those entities are 68 % fraud, versus 0.06 % for pseudo-entities seen in Train without fraud.

**Literature [L].**
- Fraud systems face concept drift and verification latency, so evaluation must be temporal [REF-004, REF-005, REF-023].
- Validation-before-test separation is standard leakage control [REF-026, REF-027].
- Prequential / rolling-origin evaluation is recommended for fraud and temporal data [REF-046, REF-024, REF-025].

## Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Random or stratified split, or k-fold cross-validation (CV) | Label propagation across linked transactions [REF-048, partially verified] and the observed entity recurrence [D] would inflate all metrics [REF-026, REF-027]. |
| **B**: as chosen, plus a 7-day gap (Train 1–98, V1 99–119, V2 120–147, gap 148–154, Test 155–182) | Loses 21,443 rows (3.6 %). The dataset contains no label-report dates, so any gap length is arbitrary [D]. Without refitting, training data already ends 49 days before Test. |
| **C**: 35-day Test (Train 1–91, V1 92–119, V2 120–147, Test 148–182) | About 27 % more Test rows (an approximately 11 % smaller detectable effect), but Train shrinks to 54.2 % and V1 grows beyond what early stopping needs. |
| Two-way split (Train / Test) with thresholds chosen on Train or Test | Thresholds would be selected on in-sample or test data, which is leakage. |
| Single validation set for both tuning and policy selection | V1 scores are *selected-on* (early stopping). Thresholds or calibrators fitted on them inherit optimism. A separate, later V2 is needed. |

## Rationale

1. **Train as large as possible, including the early high-volume regime.** The model then sees the full range of volumes.
2. **V2 has the same length as Test (28 days) and near-identical volume.** Threshold and budget transfer (research question RQ2) is then a like-for-like comparison.
3. **V1 and V2 have different jobs.** Early stopping makes V1 scores optimistic. V2 supplies out-of-sample scores for policy selection and calibration, and is closer in time to Test.
4. **Calibration on V2, not V1.** V1's prevalence (4.22 %) differs from V2's (3.39 %), and V1 is used for early stopping. Calibration matters only for R4. R0–R3 (R0 is the default 0.5 threshold; R3 is the daily top-K policy) rank and threshold on raw model scores, so calibration does not affect them: the Platt map with a positive slope is non-decreasing in the score (the input clipping and finite precision can merge extreme scores into ties), while isotonic regression is monotone but can merge scores into ties [I; REF-015, REF-016, REF-017].
5. **No primary gap.** Its purpose would be to simulate label latency [REF-005, REF-046], but its length cannot be grounded in this dataset. Because models are not refit, the Train → Test separation is already 49 days.

## Consequences

- Development data total 511,998 rows and 17,889 frauds. Test has 78,542 rows (sealed).
- Projected Test frauds, assuming V2 prevalence, are about 2,660. This is a projection, not an observation, and it drives the power discussion in the protocol.
- Daily-level Test statistics rest on **n = 28 days** (n = number of Test relative days).
- **RQ2 observes only one V2 → Test transfer event.** A development-only **rolling-origin supplement** (exploratory; inside days 1–154; primary configuration only; retrospective, as specified in the Decision) shows how variable the transfer is across three origins; it describes recurrence across these dependent historical episodes, not a general failure probability. It does not replace the single confirmatory Test evaluation. The primary configuration is the H3 configuration selected on V2 ([DR-010](DR-010-h3-confirmatory-configuration.md)).
- **Label-latency sensitivity.** Label-dependent policy choices (R1 threshold; R4 calibrator) are repeated using only V2 days 127–147. This costs no Test access.
- If the protocol is ever changed to **refit** on Train+V1(+V2), this decision must be revisited and a gap reconsidered.
- The partitions and their reproducibility anchors are declared machine-readably in [`config/dataset.yaml`](../../config/dataset.yaml). The data-preparation code (Issue #62) reproduces the row counts, `TransactionDT` bounds and hash prefixes above, and stops if any of them differs ([dataset manifest](../dataset.md#dataset-manifest)).

## Threats and limitations

- **Unavoidable optimism from entity-level label dependence.** A temporally split model can still recognise entities that already carry fraud labels. That is realistic only if those labels would have been available at scoring time, and the dataset has no report dates [REF-048, partially verified; D].
- **Episodic prevalence.** Weekly fraud rates range from 2.1 % to 5.1 %, so a single 4-week Test window may be atypical.
- **Relative time only.** No seasonal or calendar claims are possible.
- **The relative-day boundary falls during active hours** (the activity trough is 8–9 hours after it). This affects which transactions compete in the same daily top-K queue, but not model comparisons.
- **The split is a design choice.** Its defence is that it was fixed from volume alone, before any label analysis, and is recorded with partition hashes.

## References

REF-004, REF-005, REF-015, REF-016, REF-017, REF-023, REF-024, REF-025, REF-026, REF-027, REF-046, REF-047, REF-048 — see [`../references.md`](../references.md).

## Symbols and notation

| Symbol / abbreviation | Meaning |
|---|---|
| [D], [L], [I] | Evidence tags: dataset observation, literature, inference |
| CV | Cross-validation (in "k-fold CV"); k is the number of folds |
| DR | Decision Record |
| `isFraud` | Binary fraud label (1 = fraud, 0 = legitimate) |
| d_o | Origin day of the rolling-origin supplement: the first day of its threshold window (71, 85 or 99) |
| K | Fixed daily review budget (reviews per relative day), defined in DR-002 |
| n | Number of Test relative days (28) |
| N_window | Number of transactions of a threshold window (rolling-origin supplement) |
| PSI | Population Stability Index (a measure of distribution shift) |
| r | Pooled alert rate of a transfer window: its alerts over its transactions |
| R0 | Decision policy: default threshold 0.5 on the raw model score (descriptive only) |
| F1 | Harmonic mean of precision and recall |
| R1 | Decision policy: F1-optimal threshold chosen on V2 |
| R2 | Decision policy: capacity-rate threshold chosen on V2 |
| R3 | Decision policy: daily top-K review |
| R4 | Decision policy: calibrated cost-aware capped review |
| H3 | Confirmatory hypothesis 3: alert-rate transfer fidelity |
| REF-NNN | Reference identifier in `docs/references.md` |
| RQ2 | Research question 2 (policy-transfer fidelity) |
| s | Seconds |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| `TransactionDT` | Time offset in seconds from an undisclosed reference point |
| `TransactionID` | Transaction identifier (monotone in time) |
| V1, V2 | Validation 1 (relative days 106–126); Validation 2 (relative days 127–154) |
| ⌊ · ⌋ | Round down to a whole number (integer division) |
