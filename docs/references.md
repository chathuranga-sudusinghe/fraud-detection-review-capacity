# Reference Register

Master list of every source used by this research project. Other documents cite sources by **Reference ID** (e.g. `REF-005`), so each claim can be traced back to this register when writing the dissertation.

**Last verification pass:** 23 September 2026 (REF-001–REF-060); 24 September 2026 (REF-061–REF-065, Issue #7; REF-066–REF-067, Issue #8; REF-068–REF-071, Issue #14).

## Verification status

| Status | Meaning |
|---|---|
| **Verified** | Bibliographic details (authors, title, venue, volume/issue/pages, year, DOI) checked against the publisher, the Crossref DOI registry, or official proceedings. For Greenwich sources, the page or document was read directly. |
| **Partially verified** | Some details confirmed, others not. The note says what is still missing. Do not use as sole support for a central claim until upgraded. |
| **Unverified** | Not yet checked, or not accessible. **Must not be used as evidence** until checked. |

"Verified" means the **citation** is correct. The *interpretation* of each source in project documents must still be checked against the full text before it appears in the dissertation.

Initial evidence base: the reference list of `notebooks/01_dataset_feasibility_audit.ipynb`. Notebook reference numbers `[L1]`–`[L47]` and `[G1]`–`[G5]` are mapped to register IDs in the appendix.

---

## 1. Peer-reviewed literature

| ID | Citation | DOI / URL | Type | Used for | Status | Note |
|---|---|---|---|---|---|---|
| REF-001 | He, H. & Garcia, E.A. (2009) 'Learning from imbalanced data', *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263–1284. | doi:10.1109/TKDE.2008.239 | Journal article (survey) | Imbalance terminology; PR-based evaluation | Verified | Crossref |
| REF-002 | Leevy, J.L., Khoshgoftaar, T.M., Bauder, R.A. & Seliya, N. (2018) 'A survey on addressing high-class imbalance in big data', *Journal of Big Data*, 5, 42. | doi:10.1186/s40537-018-0151-6 | Journal article (survey) | Defines "high-class imbalance" as majority:minority 100:1 to 10,000:1; supports "imbalanced / low-prevalence" wording for ≈ 28:1 | Verified | Crossref; definition confirmed from the article summary |
| REF-003 | Krawczyk, B. (2016) 'Learning from imbalanced data: open challenges and future directions', *Progress in Artificial Intelligence*, 5(4), pp. 221–232. | doi:10.1007/s13748-016-0094-0 | Journal article | Difficulty depends on more than the imbalance ratio | Verified | Crossref |
| REF-004 | Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S. & Bontempi, G. (2014) 'Learned lessons in credit card fraud detection from a practitioner perspective', *Expert Systems with Applications*, 41(10), pp. 4915–4928. | doi:10.1016/j.eswa.2014.02.026 | Journal article | Alert-based / top-k evaluation; limited investigator capacity; temporal evaluation | Verified | Crossref |
| REF-005 | Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C. & Bontempi, G. (2018) 'Credit card fraud detection: a realistic modeling and a novel learning strategy', *IEEE Transactions on Neural Networks and Learning Systems*, 29(8), pp. 3784–3797. | doi:10.1109/TNNLS.2017.2736643 | Journal article | Concept drift, verification latency, alert precision; the label-latency argument | Verified | Crossref |
| REF-006 | Hand, D.J., Whitrow, C., Adams, N.M., Juszczak, P. & Weston, D. (2008) 'Performance criteria for plastic card fraud detection tools', *Journal of the Operational Research Society*, 59(7), pp. 956–962. | doi:10.1057/palgrave.jors.2602418 | Journal article | Fraud detected given a maximum number of investigations; overall-cost criterion. **Prior art for capacity-constrained evaluation** (no novelty claim) | Verified | Crossref |
| REF-007 | Correa Bahnsen, A., Aouada, D., Stojanovic, A. & Ottersten, B. (2016) 'Feature engineering strategies for credit card fraud detection', *Expert Systems with Applications*, 51, pp. 134–142. | doi:10.1016/j.eswa.2015.12.030 | Journal article | Amount-dependent cost evaluation in fraud detection | Verified | Crossref. First author's surname is "Correa Bahnsen" |
| REF-008 | Elkan, C. (2001) 'The foundations of cost-sensitive learning', *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI 2001)*, pp. 973–978. | — (IJCAI proceedings; no DOI) | Conference paper | Cost-derived decision threshold for calibrated probabilities (R4) | Verified | Semantic Scholar / DBLP record `conf/ijcai/Elkan01`; pages confirmed |
| REF-009 | Lipton, Z.C., Elkan, C. & Naryanaswamy, B. (2014) 'Optimal thresholding of classifiers to maximize F1 measure', in *Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2014)*, LNCS 8725, pp. 225–239. | doi:10.1007/978-3-662-44851-9_15 | Conference paper | Properties of F1-optimal thresholds (R1) | Verified | Crossref |
| REF-010 | Saito, T. & Rehmsmeier, M. (2015) 'The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets', *PLoS ONE*, 10(3), e0118432. | doi:10.1371/journal.pone.0118432 | Journal article | Average precision / PR-AUC as secondary metric | Verified | Crossref |
| REF-011 | Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. & Liu, T.-Y. (2017) 'LightGBM: a highly efficient gradient boosting decision tree', *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, pp. 3146–3154. | https://proceedings.neurips.cc/paper/2017 | Conference paper | LightGBM method; native missing-value and categorical handling | Verified | Semantic Scholar / DBLP `conf/nips/KeMFWCMYL17` |
| REF-012 | Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V. & Gulin, A. (2018) 'CatBoost: unbiased boosting with categorical features', *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, pp. 6639–6649. | https://proceedings.neurips.cc/paper/2018 | Conference paper | CatBoost ordered target statistics and ordered boosting; target-leakage (prediction-shift) argument | Verified | DBLP `conf/nips/ProkhorenkovaGV18`. Semantic Scholar also lists a 2017 arXiv version: cite the 2018 NeurIPS version |
| REF-013 | Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022) 'Why do tree-based models still outperform deep learning on typical tabular data?', *Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track*. | doi:10.52202/068431-0037 | Conference paper | Tree ensembles as the reference family for tabular data; no need for extra model families | Verified | Semantic Scholar / DBLP `conf/nips/GrinsztajnOV22`; DOI as reported by Semantic Scholar |
| REF-014 | Shwartz-Ziv, R. & Armon, A. (2022) 'Tabular data: deep learning is not all you need', *Information Fusion*, 81, pp. 84–90. | doi:10.1016/j.inffus.2021.11.011 | Journal article | Same as REF-013 | Verified | Crossref |
| REF-015 | Niculescu-Mizil, A. & Caruana, R. (2005) 'Predicting good probabilities with supervised learning', *Proceedings of the 22nd International Conference on Machine Learning (ICML 2005)*, pp. 625–632. | doi:10.1145/1102351.1102430 | Conference paper | Boosted-model calibration; Platt vs isotonic | Verified | Crossref |
| REF-016 | Platt, J.C. (1999) 'Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods', in Smola, A.J., Bartlett, P., Schölkopf, B. & Schuurmans, D. (eds) *Advances in Large Margin Classifiers*. Cambridge, MA: MIT Press, pp. 61–74. | — | Book chapter | Platt scaling (candidate calibrator) | Partially verified | Title, author and year confirmed (Semantic Scholar). **Editors, publisher and pages not confirmed by lookup; check a library record.** |
| REF-017 | Zadrozny, B. & Elkan, C. (2002) 'Transforming classifier scores into accurate multiclass probability estimates', *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 694–699. | doi:10.1145/775047.775151 | Conference paper | Isotonic calibration (secondary check) | Verified | Crossref |
| REF-018 | Van Calster, B., McLernon, D.J., van Smeden, M., Wynants, L. & Steyerberg, E.W. (2019) 'Calibration: the Achilles heel of predictive analytics', *BMC Medicine*, 17, 230. | doi:10.1186/s12916-019-1466-7 | Journal article | Calibration metrics (intercept, slope, Brier) | Verified | Crossref. The Crossref record lists the STRATOS group as a corporate author |
| REF-019 | van den Goorbergh, R., van Smeden, M., Timmerman, D. & Van Calster, B. (2022) 'The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression', *Journal of the American Medical Informatics Association*, 29(9), pp. 1525–1534. | doi:10.1093/jamia/ocac093 | Journal article | Imbalance corrections worsen calibration without improving discrimination; threshold shifting gives similar classification | Verified | Crossref; findings confirmed from abstract |
| REF-020 | Carriero, A., Luijken, K., de Hond, A., Moons, K.G.M., van Calster, B. & van Smeden, M. (2025) 'The harms of class imbalance corrections for machine learning based prediction models: a simulation study', *Statistics in Medicine*, 44(3–4), e10320. | doi:10.1002/sim.10320 | Journal article | Extends REF-019 to ML algorithms including XGBoost; supports H2 expectations and the exclusion of resampling | Verified | Crossref |
| REF-021 | Dal Pozzolo, A., Caelen, O., Johnson, R.A. & Bontempi, G. (2015) 'Calibrating probability with undersampling for unbalanced classification', *2015 IEEE Symposium Series on Computational Intelligence (SSCI)*, pp. 159–166. | doi:10.1109/SSCI.2015.33 | Conference paper | Probability distortion from rebalancing in fraud data | Verified | Crossref |
| REF-022 | Chawla, N.V., Bowyer, K.W., Hall, L.O. & Kegelmeyer, W.P. (2002) 'SMOTE: synthetic minority over-sampling technique', *Journal of Artificial Intelligence Research*, 16, pp. 321–357. | doi:10.1613/jair.953 | Journal article | Defines the excluded method (context only) | Verified | Crossref |
| REF-023 | Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014) 'A survey on concept drift adaptation', *ACM Computing Surveys*, 46(4), 44. | doi:10.1145/2523813 | Journal article (survey) | Concept drift; temporal evaluation | Verified | Crossref |
| REF-024 | Tashman, L.J. (2000) 'Out-of-sample tests of forecasting accuracy: an analysis and review', *International Journal of Forecasting*, 16(4), pp. 437–450. | doi:10.1016/S0169-2070(00)00065-0 | Journal article | Rolling-origin evaluation | Verified | Crossref |
| REF-025 | Cerqueira, V., Torgo, L. & Mozetič, I. (2020) 'Evaluating time series forecasting models: an empirical study on performance estimation methods', *Machine Learning*, 109(11), pp. 1997–2028. | doi:10.1007/s10994-020-05910-7 | Journal article | Out-of-sample / rolling-origin estimation under temporal dependence | Verified | Crossref |
| REF-026 | Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012) 'Leakage in data mining: formulation, detection, and avoidance', *ACM Transactions on Knowledge Discovery from Data*, 6(4), 15. | doi:10.1145/2382577.2382579 | Journal article | Leakage taxonomy | Verified | Crossref |
| REF-027 | Kapoor, S. & Narayanan, A. (2023) 'Leakage and the reproducibility crisis in machine-learning-based science', *Patterns*, 4(9), 100804. | doi:10.1016/j.patter.2023.100804 | Journal article | Leakage taxonomy (temporal, preprocessing, non-independence) | Verified | Crossref |
| REF-028 | Dietterich, T.G. (1998) 'Approximate statistical tests for comparing supervised classification learning algorithms', *Neural Computation*, 10(7), pp. 1895–1923. | doi:10.1162/089976698300017197 | Journal article | Paired tests incl. McNemar for classifier comparison | Verified | Crossref |
| REF-029 | Holm, S. (1979) 'A simple sequentially rejective multiple test procedure', *Scandinavian Journal of Statistics*, 6(2), pp. 65–70. | doi:10.2307/4615733 (JSTOR) | Journal article | Multiplicity control for confirmatory tests | Verified | Semantic Scholar record with JSTOR DOI |
| REF-030 | Lakens, D. (2017) 'Equivalence tests: a practical primer for t tests, correlations, and meta-analyses', *Social Psychological and Personality Science*, 8(4), pp. 355–362. | doi:10.1177/1948550617697177 | Journal article | TOST equivalence testing; SESOI (H2, H3) | Verified | Crossref |
| REF-031 | Rabanser, S., Günnemann, S. & Lipton, Z.C. (2019) 'Failing loudly: an empirical study of methods for detecting dataset shift', *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, pp. 1394–1406. | https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html | Conference paper | Background on detecting distribution shift | Verified | NeurIPS proceedings / DBLP `conf/nips/RabanserGL19` |
| REF-032 | Rodolfa, K.T., Lamba, H. & Ghani, R. (2021) 'Empirical observation of negligible fairness–accuracy trade-offs in machine learning for public policy', *Nature Machine Intelligence*, 3(10), pp. 896–904. | doi:10.1038/s42256-021-00396-x | Journal article | Example of top-k evaluation under resource constraints outside fraud | Verified | Crossref |
| REF-033 | Boyd, S., Cortes, C., Mohri, M. & Radovanovic, A. (2012) 'Accuracy at the top', *Advances in Neural Information Processing Systems 25 (NIPS 2012)*, pp. 953–961. | https://proceedings.neurips.cc/paper/2012/hash/7fe1f8abaad094e0b5cb1b01d712f708-Abstract.html | Conference paper | Formal treatment of accuracy in the top quantile (top-K policies) | Verified | NeurIPS proceedings page |
| REF-034 | Wilson, E.B. (1927) 'Probable inference, the law of succession, and statistical inference', *Journal of the American Statistical Association*, 22(158), pp. 209–212. | doi:10.1080/01621459.1927.10502953 | Journal article | Wilson score intervals for proportions | Verified | Crossref |
| REF-035 | McNemar, Q. (1947) 'Note on the sampling error of the difference between correlated proportions or percentages', *Psychometrika*, 12(2), pp. 153–157. | doi:10.1007/BF02295996 | Journal article | McNemar test (secondary) | Verified | Crossref |
| REF-036 | Field, C.A. & Welsh, A.H. (2007) 'Bootstrapping clustered data', *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 69(3), pp. 369–390. | doi:10.1111/j.1467-9868.2007.00593.x | Journal article | Cluster bootstrap (fraud clustering by entity) | Verified | Crossref |
| REF-037 | Künsch, H.R. (1989) 'The jackknife and the bootstrap for general stationary observations', *The Annals of Statistics*, 17(3), pp. 1217–1241. | doi:10.1214/aos/1176347265 | Journal article | Block bootstrap (day-block resampling) | Partially verified | Crossref confirms authors, title, volume and issue. **The page range is not in the Crossref record; check it on the publisher page.** |
| REF-038 | Hernández-Orallo, J., Flach, P. & Ferri, C. (2012) 'A unified view of performance metrics: translating threshold choice into expected classification loss', *Journal of Machine Learning Research*, 13, pp. 2813–2869. | doi:10.5555/2503308.2503332 (ACM DL record) | Journal article | Threshold-choice methods as part of evaluation | Verified | Semantic Scholar / DBLP `journals/jmlr/Hernandez-OralloFF12` |
| REF-039 | Provost, F. & Fawcett, T. (2001) 'Robust classification for imprecise environments', *Machine Learning*, 42(3), pp. 203–231. | doi:10.1023/A:1007601015854 | Journal article | Evaluation when costs are uncertain; motivates cost-ratio sensitivity | Verified | Crossref |
| REF-040 | Drummond, C. & Holte, R.C. (2006) 'Cost curves: an improved method for visualizing classifier performance', *Machine Learning*, 65(1), pp. 95–130. | doi:10.1007/s10994-006-8199-5 | Journal article | Cost curves / break-even presentation (RQ3) | Verified | Crossref |
| REF-061 | Schuirmann, D.J. (1987) 'A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability', *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), pp. 657–680. | doi:10.1007/BF01068419 | Journal article | Original TOST procedure; equivalence with the 90 % CI inside the bounds (H3) | Verified | Crossref (added for Issue #7) |
| REF-062 | Berger, R.L. & Hsu, J.C. (1996) 'Bioequivalence trials, intersection-union tests and equivalence confidence sets', *Statistical Science*, 11(4), pp. 283–319. | doi:10.1214/ss/1032280304 | Journal article | TOST as an intersection-union test (each one-sided test at α; no α split); link to equivalence confidence sets (H3) | Verified | Crossref (volume, issue, DOI) and Semantic Scholar (pages) (added for Issue #7) |
| REF-063 | Lakens, D., Scheel, A.M. & Isager, P.M. (2018) 'Equivalence testing for psychological research: a tutorial', *Advances in Methods and Practices in Psychological Science*, 1(2), pp. 259–269. | doi:10.1177/2515245918770963 | Journal article (tutorial) | Equivalence bounds as a smallest effect size of interest (SESOI) that the researcher must justify (H3; H2 SESOI) | Verified | Crossref. **Only the abstract was read; the full text returned HTTP 403.** Cite only for the TOST/SESOI framing stated in the abstract until the full text is checked (added for Issue #7) |
| REF-064 | Hall, P., Horowitz, J.L. & Jing, B.-Y. (1995) 'On blocking rules for the bootstrap with dependent data', *Biometrika*, 82(3), pp. 561–574. | doi:10.1093/biomet/82.3.561 | Journal article | Order of the optimal block length (n^(1/3) for variance, n^(1/4) for one-sided distribution estimation); H3 block-length choice | Verified | Crossref; the block-length orders are confirmed from the abstract and summary (added for Issue #7) |
| REF-066 | Jaeschke, R., Singer, J. & Guyatt, G.H. (1989) 'Measurement of health status: ascertaining the minimal clinically important difference', *Controlled Clinical Trials*, 10(4), pp. 407–415. | doi:10.1016/0197-2456(89)90005-6 | Journal article | Origin of the minimal clinically important difference: a practically meaningful difference defined by its consequences rather than by statistical significance. Principle adapted for the H1/H2 SESOI (no clinical value used) | Verified | Crossref; the definition was confirmed from secondary quotations of the paper. Full text not read (added for Issue #8) |
| REF-067 | Benavoli, A., Corani, G., Demšar, J. & Zaffalon, M. (2017) 'Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis', *Journal of Machine Learning Research*, 18(77), pp. 1–36. | https://www.jmlr.org/papers/v18/16-305.html | Journal article (tutorial) | ML precedent for an analyst-chosen region of practical equivalence (±0.01 accuracy) when comparing classifiers; critique of significance without magnitude. **Not** used as the source of the SESOI value | Verified | JMLR PDF read; the ±0.01 accuracy ROPE was confirmed in the text (added for Issue #8) |
| REF-068 | Owen, A.B. (2007) 'The pigeonhole bootstrap', *The Annals of Applied Statistics*, 1(2), pp. 386–411. | doi:10.1214/07-AOAS122 | Journal article | Two-way (row × column) bootstrap for crossed, unbalanced random-effects data; mean consistency; mildly conservative variance, with an optional correction (subtract 2 × naive variance). Primary H1/H2 method and S1 (DR-006) | Verified | Crossref (volume, issue, DOI); pages from the Project Euclid record; the conservativeness and correction statements were read in the arXiv full text (0712.1111) (added for Issue #14) |
| REF-069 | Cameron, A.C., Gelbach, J.B. & Miller, D.L. (2008) 'Bootstrap-based improvements for inference with clustered errors', *The Review of Economics and Statistics*, 90(3), pp. 414–427. | doi:10.1162/rest.90.3.414 | Journal article | Standard cluster-robust inference can over-reject with few (five to thirty) clusters; relevant to the 28-day Test period (DR-006) | Verified | Crossref; the few-cluster finding was confirmed from the abstract. Full text not read (added for Issue #14) |
| REF-070 | Cameron, A.C., Gelbach, J.B. & Miller, D.L. (2011) 'Robust inference with multiway clustering', *Journal of Business & Economic Statistics*, 29(2), pp. 238–249. | doi:10.1198/jbes.2010.07136 | Journal article | Inference must account for non-nested, multi-way clustering (here: day × entity) (DR-006) | Verified | Crossref; the scope was confirmed from the abstract (RePEc). Full text not read (added for Issue #14) |
| REF-071 | Obuchowski, N.A. (1998) 'On the comparison of correlated proportions for clustered data', *Statistics in Medicine*, 17(13), pp. 1495–1507. | doi:10.1002/(SICI)1097-0258(19980715)17:13<1495::AID-SIM863>3.0.CO;2-I | Journal article | Under intracluster correlation, the size of McNemar's test can greatly exceed the nominal level. Keeps McNemar secondary and descriptive (DR-005, DR-006) | Verified | Crossref and PubMed (PMID 9695194); the statement was confirmed from the abstract (added for Issue #14) |

### 1.1 Studies using the IEEE-CIS dataset (prior-work context)

| ID | Citation | DOI / URL | Type | Used for | Status | Note |
|---|---|---|---|---|---|---|
| REF-041 | Kang, J. & Kim, K. (2026) 'Deployment-oriented interpretable fraud detection via hybrid explainable boosting machines with concept–raw fusion on the IEEE-CIS benchmark', *Applied Sciences*, 16(12), 5809. | doi:10.3390/app16125809 | Journal article | Evidence that chronological out-of-time evaluation of IEEE-CIS already exists, so no novelty claim is made for temporal evaluation | Partially verified | Crossref confirms the bibliographic details. **The full text could not be retrieved (HTTP 403); the claimed split design and CatBoost comparator come only from a search abstract. Read before citing.** |
| REF-042 | Meena, R., Wadhwani, R. & Rasool, A. (2026) 'A comparative review of fraud detection techniques using the IEEE-CIS dataset', *Lecture Notes in Networks and Systems*, Springer, pp. 207–223. | doi:10.1007/978-3-032-10940-8_17 | Book chapter (review) | Context: the dataset is widely used | Partially verified | Crossref confirms authors, title and pages. **Book title, volume number and full text not checked.** |

## 2. Methodological books and handbook sources

| ID | Citation | DOI / URL | Type | Used for | Status | Note |
|---|---|---|---|---|---|---|
| REF-043 | Fernández, A., García, S., Galar, M., Prati, R.C., Krawczyk, B. & Herrera, F. (2018) *Learning from Imbalanced Data Sets*. Cham: Springer. | doi:10.1007/978-3-319-98074-4 | Book | Imbalance terminology and difficulty factors | Verified | Crossref |
| REF-044 | Efron, B. & Tibshirani, R.J. (1993) *An Introduction to the Bootstrap*. New York: Chapman & Hall. | doi:10.1201/9780429246593 | Book | Bootstrap confidence intervals | Verified | Crossref record (Chapman & Hall/CRC; digital edition dated 1994). Cite the 1993 first edition |
| REF-045 | Siddiqi, N. (2006) *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Hoboken, NJ: Wiley. | doi:10.1002/9781119201731 | Practitioner book | Conventional Population Stability Index (PSI) bands 0.10 / 0.25 (heuristics only) | Partially verified | The DOI resolves to the Wiley online edition (dated 2012). **The print year and the page with the PSI bands are not checked.** Use only for the heuristic PSI bands, not as statistical justification |
| REF-046 | Le Borgne, Y.-A., Siblini, W., Lebichot, B. & Bontempi, G. (2022) *Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook*. Université Libre de Bruxelles. | https://fraud-detection-handbook.github.io/fraud-detection-handbook/ | Open handbook (not peer-reviewed) | Train/delay/test periods; prequential (rolling) validation; top-k and card-level precision | Partially verified | The validation-strategies chapter was read (7-day train/delay/test example; prequential validation recommended). The site now sits behind a bot check. **Authorship and year not confirmed against a formal record.** |

## 3. Dataset and organiser sources

| ID | Citation | DOI / URL | Type | Used for | Status | Note |
|---|---|---|---|---|---|---|
| REF-047 | IEEE Computational Intelligence Society & Vesta Corporation (2019) *IEEE-CIS Fraud Detection* [competition data]. Kaggle. | https://www.kaggle.com/competitions/ieee-fraud-detection/data | Dataset / organiser documentation | Dataset source; `isFraud` target; `TransactionDT` as a relative time delta; table structure | Partially verified | The raw files are held locally (SHA-256 values in the feasibility notebook, §2.1). **The data page requires a Kaggle login and was not read directly in this pass.** |
| REF-048 | Vesta Corporation (2019) 'Data Description (Details and Discussion)', IEEE-CIS Fraud Detection competition discussion forum, thread 101203. Kaggle. | https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203 | Organiser forum post (not peer-reviewed) | Labelling rule: chargeback, plus linked later transactions labelled fraud, 120-day window; amounts in USD; feature-group descriptions | Partially verified | The labelling rule was confirmed from search-result extracts of this thread. **Read the full post while logged in and record the exact quotations before relying on it in the dissertation.** |
| REF-049 | Kaggle (2019) *IEEE-CIS Fraud Detection — Competition Rules*. | https://www.kaggle.com/competitions/ieee-fraud-detection/rules | Data-use terms | Permitted use of the competition data for academic research; redistribution limits | **Unverified** | Not accessed (login required). **Open item before protocol freeze.** |

## 4. University of Greenwich guidance

All accessed 23 September 2026.

| ID | Citation | URL | Type | Used for | Status | Note |
|---|---|---|---|---|---|---|
| REF-050 | University of Greenwich (2026) *Research Ethics Policy* (February 2026 version). | https://www.gre.ac.uk/docs/rep/gre/research-ethics-policy (PDF: https://www.gre.ac.uk/__data/assets/pdf_file/0041/399884/research-ethics-policy-february-2026.pdf) | University policy | Scope incl. master's students (§1.2); non-anonymised personal information (§1.3.1 iii); harmful/criminal subject matter (§1.3.4); Higher-Risk categorisation (§2.2.1 iv); FREC delegation (§2.3a); UREB referral (§2.4, §2.5.2) | Verified | PDF text extracted and the relevant sections read |
| REF-051 | University of Greenwich (n.d.) *Guidance on Ethical Approval for Research*. | https://www.gre.ac.uk/research/governance-and-awards/research-ethics-committee/guidance-on-ethical-approval-for-research | University guidance | Anonymised secondary data; taught students advised by the programme/module leader; secure (encrypted) data storage | Verified | Page read; no publication date shown |
| REF-052 | University of Greenwich (n.d.) *Research Integrity*. | https://www.gre.ac.uk/research/governance-and-awards/research-integrity | University guidance | Commitment to the Concordat to Support Research Integrity (2025) | Verified | Page read |
| REF-053 | University of Greenwich (n.d.) *AI for students*. | https://www.gre.ac.uk/ai-guidance/students | University guidance | AI Reflection (AIR) declaration; three-tier AI-use framework | Verified | Page read; no publication date shown |
| REF-054 | University of Greenwich (n.d.) *Data Science, MSc* [course page]. | https://www.gre.ac.uk/postgraduate-courses/engsci/computer-science-data-science-msc | Course information | General description of the MSc project | Verified | Page read. It contains no COMP1885 module detail |
| REF-055 | University of Greenwich (2026) *COMP1885 Individual Project — module handbook and marking criteria*. | Moodle (not public) | Module documentation | Required dissertation structure, ethics deadlines, marking criteria, AI-use tier | **Unverified** | Not publicly accessible. **Obtain from Moodle or the module leader.** |

## 5. Other authoritative sources

| ID | Citation | DOI / URL | Type | Used for | Status | Note |
|---|---|---|---|---|---|---|
| REF-065 | European Medicines Agency, Committee for Medicinal Products for Human Use (2010) *Guideline on the investigation of bioequivalence*, CPMP/EWP/QWP/1401/98 Rev. 1/Corr\*\*. London, 20 January 2010; effective 1 August 2010. | https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-investigation-bioequivalence-rev1_en.pdf | Regulatory guideline | Precedent that equivalence margins for positive quantities are defined relative to a reference (the 90 % CI of the test/reference ratio must lie within 80.00–125.00 %). **Used for the form of the H3 margin only, not its magnitude** | Verified | PDF read; the 80.00–125.00 % acceptance interval and the 90 % CI rule were confirmed in the text (added for Issue #7) |

Practitioner web pages (e.g. vendor blogs, unreviewed Kaggle notebooks) and non-peer-reviewed preprints are deliberately excluded.

## 6. Sources listed in `docs/literature-review.md` awaiting verification

These appear in the literature-review working document **without authors or DOIs**. They have **not been checked** and must not be cited as evidence until they are.

| ID | Source as currently listed | Status | Required action |
|---|---|---|---|
| REF-056 | "Credit card fraud detection in the era of disruptive technologies: A systematic review", *Journal of King Saud University – Computer and Information Sciences*, 2023 | **Unverified** | Confirm authors, volume, pages, DOI |
| REF-057 | "Threshold optimization and random undersampling for imbalanced credit card data", *Journal of Big Data*, 2023 | **Unverified** | Confirm authors, article number, DOI |
| REF-058 | "TEMPLAR fraud: a verifier-grounded, calibrated and cost-sensitive framework for transaction fraud detection", *Discover Data*, 2026 | **Unverified** | Confirm existence and bibliographic details |
| REF-059 | "Explainable ensemble machine learning for financial transaction fraud detection: Insights from XGBoost and deep neural networks", 2026 | **Unverified** | Venue unknown; confirm or remove |
| REF-060 | "DISCO: Decoupling representation learning and risk control for reliable credit card fraud detection", *Decision Support Systems*, 2026 | **Unverified** | Confirm authors, volume, DOI |

---

## Appendix — mapping from feasibility-notebook reference numbers

| Notebook | Register | Notebook | Register | Notebook | Register |
|---|---|---|---|---|---|
| L1 | REF-001 | L17 | REF-015 | L33 | REF-030 |
| L2 | REF-002 | L18 | REF-016 | L34 | REF-031 |
| L3 | REF-003 | L19 | REF-017 | L35 | REF-045 |
| L4 | REF-043 | L20 | REF-018 | L36 | REF-032 |
| L5 | REF-004 | L21 | REF-019 | L37 | REF-033 |
| L6 | REF-005 | L22 | REF-020 | L38 | REF-047, REF-048 |
| L7 | REF-006 | L23 | REF-021 | L39 | REF-034 |
| L8 | REF-046 | L24 | REF-022 | L40 | REF-035 |
| L9 | REF-007 | L25 | REF-023 | L41 | REF-036 |
| L10 | REF-008 | L26 | REF-024 | L42 | REF-037 |
| L11 | REF-009 | L27 | REF-025 | L43 | REF-038 |
| L12 | REF-010 | L28 | REF-026 | L44 | REF-039 |
| L13 | REF-011 | L29 | REF-027 | L45 | REF-040 |
| L14 | REF-012 | L30 | REF-028 | L46 | REF-041 |
| L15 | REF-013 | L31 | REF-044 | L47 | REF-042 |
| L16 | REF-014 | L32 | REF-029 | G1–G5 | REF-050–REF-054 |

**Corrections relative to the notebook's reference list:**
- L46 and L47 now have confirmed authors (Kang & Kim; Meena, Wadhwani & Rasool) and year 2026.
- L9's first author is "Correa Bahnsen".
- L42's page range is pending (REF-037).
- L18 is only partially verified (REF-016).

The notebook text itself is unaffected: it already flags L46/L47 for verification.
