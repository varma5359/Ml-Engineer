# Day 54: Gradient Boosting Fundamentals

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Beginner to Intermediate  
**Time:** 2–3 hours  
**Approved mini-project:** Boosting Model Comparison

## 1. Learning objectives
- Explain ensemble learning, bagging and boosting.
- Describe how gradient boosting builds learners sequentially.
- Identify the role of loss, residuals, learning rate and estimators.
- Train and compare a decision tree, Random Forest and Gradient Boosting model.

## 2. What is gradient boosting?
Gradient boosting is an ensemble method that builds a sequence of small models. Each new model is trained to improve the current ensemble by reducing the remaining loss.

**Flow:** Initial prediction → calculate errors/gradients → train a small tree → add a scaled correction → repeat.

For squared-error regression, residuals correspond to the negative gradient of the loss.

## 3. Bagging versus boosting

| Bagging | Boosting |
|---|---|
| Learners are generally trained independently | Learners are trained sequentially |
| Combines predictions to reduce variance | Each stage improves the current ensemble |
| Models can often be trained in parallel | Later stages depend on earlier stages |
| Random Forest | Gradient Boosting |

Both methods can overfit. Performance depends on data, complexity and validation.

## 4. Important parameters

| Parameter | Meaning |
|---|---|
| `n_estimators` | Number of boosting stages |
| `learning_rate` | Scales each new learner's contribution |
| `max_depth` | Limits tree depth |
| `min_samples_leaf` | Minimum observations in a leaf |
| `subsample` | Fraction of training rows used per stage |

A lower learning rate often needs more estimators. Tune parameters with validation data rather than assuming a fixed combination is best.

## 5. Practical example — regression

### 5.1 Create reproducible data
```python
import pandas as pd
from sklearn.datasets import make_regression

X_values, y = make_regression(
    n_samples=800, n_features=6, n_informative=4,
    noise=18, random_state=42
)
X = pd.DataFrame(X_values, columns=[f"feature_{i}" for i in range(1, 7)])
print(X.head())
```

**Expected output:** Five rows with six feature columns. This is synthetic teaching data.

### 5.2 Split the data
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape)
```

**Expected output:** 640 training rows and 160 test rows.

### 5.3 Train three models
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05,
        max_depth=2, random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "trained")
```

### 5.4 Compare performance
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rows = []
for name, model in models.items():
    pred = model.predict(X_test)
    rows.append({
        "model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred)
    })

print(pd.DataFrame(rows).sort_values("RMSE"))
```

**Interpretation:** Lower MAE/RMSE and higher R² are generally preferred. Also consider runtime, stability and business costs. Exact values depend on the environment and data.

## 6. Classification variation
Use `GradientBoostingClassifier` for classification. Evaluate confusion matrix, precision, recall, F1 and ROC-AUC or PR-AUC where appropriate. Accuracy alone can be misleading for imbalanced targets.

## 7. Approved mini-project — Boosting Model Comparison

**Scenario:** A retailer wants to estimate weekly product demand from historical attributes.

**Tasks**
1. Load a numeric-target dataset and inspect its quality.
2. Create one reproducible train/test split.
3. Train a Decision Tree, Random Forest and Gradient Boosting model.
4. Compare MAE, RMSE and R².
5. Record parameters and training time.
6. Explain the trade-offs and recommend a candidate.

**Suggested structure**
```text
boosting_model_comparison/
├── data/
├── notebooks/model_comparison.ipynb
├── reports/comparison.md
└── README.md
```

**Validation:** Use the same split for all models, remove leakage, evaluate held-out data and avoid claiming feature importance is causal.

## 8. Common mistakes
- Assuming boosting always wins.
- Using deep trees at every stage without validation.
- Increasing estimators without checking generalization.
- Comparing models on different splits.
- Reporting only training scores.
- Treating residuals as explanations of causes.

## 9. Quick revision
- Boosting is sequential ensemble learning.
- Gradient refers to the direction that reduces loss.
- Learning rate scales each stage.
- Estimators control the number of stages.
- Shallow trees help control learner complexity.
- Always compare with a baseline.

## 10. Completion checklist
- [ ] Explain bagging versus boosting.
- [ ] Describe the gradient boosting process.
- [ ] Train GradientBoostingRegressor.
- [ ] Compare three models consistently.
- [ ] Complete Boosting Model Comparison.

**Next lesson:** Day 55 — XGBoost and practical boosting workflows.
