# CausalCare: Methodology

## Causal Framework

This project uses the **potential outcomes framework** (Rubin, 1974) to estimate the Average Treatment Effect (ATE) of early beta-blocker administration on hospital mortality. The fundamental problem of causal inference — we can never observe both potential outcomes for the same patient — motivates the use of multiple estimation strategies.

## Identification Strategy

Under the **backdoor criterion** (Pearl, 2009), the causal effect is identifiable if we condition on a sufficient set of pre-treatment confounders that block all backdoor paths from treatment to outcome. Our DAG encodes the assumption that the measured confounders (demographics, APACHE severity, labs, comorbidities, concurrent treatments) satisfy this criterion.

### Key Assumptions

1. **Conditional ignorability (no unmeasured confounders):** Given measured confounders W, treatment assignment is independent of potential outcomes. This is the strongest and least testable assumption.
2. **Positivity:** Every patient has a non-zero probability of receiving either treatment level, conditional on confounders. Assessed via propensity score overlap plots.
3. **Consistency:** The observed outcome under treatment T=t equals the potential outcome Y(t). Requires a well-defined treatment.
4. **No interference (SUTVA):** One patient's treatment does not affect another's outcome.

## Estimation Methods

### Propensity Score Matching
1:1 nearest-neighbor matching on the logit of the propensity score with a caliper of 0.2 standard deviations. Propensity scores estimated via 5-fold cross-validated logistic regression to prevent overfitting.

### Inverse Probability Weighting (IPW)
Stabilized weights with trimming at the 1st and 99th percentiles to reduce variance from extreme weights. Effective sample size reported as a diagnostic.

### Augmented IPW (Doubly Robust)
Cross-fitted AIPW estimator that combines a propensity score model with separate outcome models for treated and control groups. Consistent if either model is correctly specified.

### Double Machine Learning (DML)
LinearDML from EconML with gradient boosted trees as nuisance estimators. Uses cross-fitting (5-fold) to avoid regularization bias. Provides valid asymptotic inference.

### Causal Forest
CausalForestDML from EconML for heterogeneous treatment effect (CATE) estimation. Enables subgroup analysis to identify which patients benefit most from early beta-blocker therapy.

## Sensitivity Analysis

### E-values
Quantifies the minimum strength of association an unmeasured confounder would need with both treatment and outcome to fully explain away the observed effect (VanderWeele & Ding, 2017).

### DoWhy Refutation Tests
- **Placebo treatment:** Permutes treatment labels; effect should vanish
- **Random common cause:** Adds noise confounder; estimate should be stable
- **Data subset:** Re-estimates on 80% subsets; estimate should be robust
- **Unobserved common cause:** Simulates an unmeasured confounder with specified effect strengths

## References

- Hernán MA, Robins JM. *Causal Inference: What If.* Chapman & Hall/CRC, 2020.
- Pearl J. *Causality.* Cambridge University Press, 2009.
- VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. *Ann Intern Med.* 2017;167(4):268–274.
- Chernozhukov V, et al. Double/debiased machine learning for treatment and structural parameters. *Econom J.* 2018;21(1):C1–C68.
- Athey S, Tibshirani J, Wager S. Generalized random forests. *Ann Stat.* 2019;47(2):1148–1178.
