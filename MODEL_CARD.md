# Model Card: Telco Customer Churn Classifier (v1.0.0)

## 1. Model Details
- **Architecture:** Scikit-Learn `Pipeline` (`NumericStringCleaner` -> `ColumnTransformer` [Median Imputer + `StandardScaler` for numeric; Constant Imputer + `OneHotEncoder(handle_unknown="ignore")` for categorical] -> `GradientBoostingClassifier`).
- **Tuning:** 5-fold `StratifiedKFold` `RandomizedSearchCV` optimizing Precision-Recall AUC (`average_precision`).
- **Traceability:** Every prediction returns `model_version` and the SHA-256 hash of the exact training dataframe (`training_data_sha256`).

## 2. Intended Use
- **Primary User:** Telecom CRM Retention Engine.
- **Decision Workflow:** Customers scored above the optimal decision threshold are routed to automated retention workflows (proactive outreach or contract renewal incentives).
- **Out-of-Scope:** Automated involuntary account termination or credit underwriting.

## 3. Threshold Justification (Cost-Sensitive Optimization)
Default `0.50` probability thresholds assume false positives and false negatives carry equal cost, which fails in subscription businesses with modest class imbalance (~26.5% churn):
- **False Negative Cost ($C_{FN} = \$500$):** Missing an active churner forfeits ~12 months of gross margin (~$65/mo ARPU x 65% margin).
- **False Positive Cost ($C_{FP} = \$50$):** Offering a retention incentive (e.g., $10/mo discount for 5 months or a loyalty perk) to a customer who would have stayed.

Because $C_{FN} : C_{FP} = 10 : 1$, sweeping thresholds across validation probabilities minimizes total expected cost:

$$\text{Expected Cost}(t) = \frac{500 \cdot \text{FN}(t) + 50 \cdot \text{FP}(t)}{N}$$

This shifts the optimal decision threshold downward (typically into the **`0.22 - 0.30`** range), capturing ~80%+ of at-risk churners while keeping campaign ROI positive.

## 4. Limitations & Fairness Considerations
- Trained on historical snapshot data; sudden competitor pricing shocks or network outages require retraining.
- `SeniorCitizen` and `gender` are included as baseline control features; retention offers must never penalize protected demographics (offers are strictly positive incentives).
