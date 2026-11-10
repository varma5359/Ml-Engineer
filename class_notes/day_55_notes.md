# Day 55: XGBoost and Practical Boosting Workflows

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Time:** 2–3 hours  
**Approved mini-project:** Customer Churn Boosting Model

## 1. Learning objectives
- Explain XGBoost and its advantages.
- Identify important XGBoost parameters.
- Train an XGBoost classifier.
- Evaluate churn predictions using class-aware metrics.
- Explain threshold and imbalance trade-offs.

## 2. What is XGBoost?
XGBoost means **Extreme Gradient Boosting**. It is an optimized gradient-boosted tree library with regularization and efficient training techniques.

It supports classification and regression, provides missing-value handling in tree learning, and offers row/column sampling and regularization controls. It is not automatically the best model for every dataset.

## 3. Important parameters

| Parameter | Purpose |
|---|---|
| `n_estimators` | Number of boosting rounds |
| `learning_rate` / `eta` | Contribution per round |
| `max_depth` | Maximum tree depth |
| `min_child_weight` | Minimum child weight required for a split |
| `subsample` | Fraction of rows sampled |
| `colsample_bytree` | Fraction of features sampled |
| `reg_alpha` | L1 regularization |
| `reg_lambda` | L2 regularization |
| `objective` | Learning objective |
| `eval_metric` | Evaluation metric |

Defaults and supported options can vary by version. Check the installed version's documentation.

## 4. Install and verify
```bash
python -m pip install xgboost scikit-learn pandas
```

```python
import xgboost
print(xgboost.__version__)
```

## 5. Churn classification workflow

### 5.1 Create synthetic teaching data
```python
import pandas as pd
from sklearn.datasets import make_classification

X_values, y = make_classification(
    n_samples=1200, n_features=8, n_informative=5,
    n_redundant=1, weights=[0.75, 0.25], random_state=42
)
columns = [
    "tenure_months", "monthly_charges", "support_calls",
    "late_payments", "usage_change", "contract_code",
    "service_count", "discount"
]
X = pd.DataFrame(X_values, columns=columns)
y = pd.Series(y, name="churn")
print(y.value_counts(normalize=True))
```

**Explanation:** Synthetic labels and illustrative feature names are used; they are not evidence of real churn drivers.

### 5.2 Split using stratification
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(y_train.mean(), y_test.mean())
```

### 5.3 Train XGBoost
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)
print("Training completed")
```

### 5.4 Evaluate
```python
from sklearn.metrics import confusion_matrix, classification_report

predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
```

**Interpretation:** Focus on churn-class precision and recall. False negatives miss customers who may leave; false positives may trigger unnecessary retention costs.

### 5.5 Inspect probability and threshold
```python
probabilities = model.predict_proba(X_test)[:, 1]
custom_predictions = (probabilities >= 0.35).astype(int)
print(classification_report(y_test, custom_predictions, zero_division=0))
```

The 0.35 threshold is illustrative only. Select thresholds on validation data, not the final test set.

## 6. Imbalanced data considerations
- Inspect class counts.
- Use stratified splitting where suitable.
- Consider class weighting or `scale_pos_weight`.
- Evaluate precision, recall, F1 and PR-AUC.
- Do not resample before splitting.
- If using SMOTE, apply it only within training folds/pipelines.

## 7. Approved mini-project — Customer Churn Boosting Model

**Scenario:** A subscription business wants to prioritize retention outreach.

**Tasks**
1. Load a public or synthetic churn dataset.
2. Identify target and prediction horizon.
3. Remove identifiers unless they are justified features.
4. Inspect missing values and class balance.
5. Create a leakage-safe stratified or time-based split.
6. Train a baseline and XGBoost model.
7. Compare confusion matrix, precision, recall, F1 and PR-AUC where suitable.
8. Select a threshold using validation evidence.
9. Document trade-offs and limitations.

**Suggested structure**
```text
customer_churn_boosting/
├── data/
├── notebooks/
├── src/data_prep.py
├── src/train.py
├── reports/evaluation.md
└── README.md
```

**Submission:** Metrics table, parameter summary, threshold rationale and reproducible code.

## 8. Common mistakes
- Optimizing accuracy on imbalanced data.
- Using test data to choose a threshold.
- Applying SMOTE before splitting.
- Including IDs that leak target information.
- Reporting only training scores.
- Claiming feature importance explains causation.

## 9. Completion checklist
- [ ] Explain XGBoost parameters.
- [ ] Train an XGBClassifier.
- [ ] Evaluate class-specific metrics.
- [ ] Explain threshold trade-offs.
- [ ] Complete Customer Churn Boosting Model.

**Next lesson:** Day 56 — LightGBM and CatBoost concepts and use cases.
