# Day 56: LightGBM and CatBoost Concepts and Use Cases

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Time:** 2–3 hours  
**Approved mini-project:** Tabular Model Comparison

## 1. Learning objectives
- Explain LightGBM and CatBoost.
- Describe histogram-based and leaf-wise growth.
- Explain CatBoost categorical handling.
- Train and compare boosting libraries on tabular data.
- Select a model using performance and operational trade-offs.

## 2. Why different boosting libraries?
Tabular datasets vary in size, feature types, missing values and compute constraints. Libraries make different engineering and algorithmic choices. No library is universally best.

## 3. LightGBM
LightGBM is designed for efficient gradient boosting, particularly on tabular datasets.

- **Histogram-based split finding:** continuous values are grouped into bins to reduce split-search cost.
- **Leaf-wise growth:** expands the leaf with the greatest estimated loss reduction.
- Leaf-wise growth can improve loss efficiently but may overfit on small data without constraints.

Important parameters:

| Parameter | Meaning |
|---|---|
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Contribution per round |
| `num_leaves` | Maximum leaves per tree |
| `max_depth` | Depth limit |
| `min_child_samples` | Minimum rows in a leaf |
| `subsample` | Row sampling fraction |
| `colsample_bytree` | Feature sampling fraction |

## 4. CatBoost
CatBoost is a gradient-boosting library designed to handle categorical features effectively.

- It provides built-in categorical feature processing.
- Ordered boosting is designed to reduce certain prediction-shift issues during training.
- It can be useful when a dataset contains many categorical columns.
- Categorical columns must be declared and consistently supplied at inference.

Important parameters:

| Parameter | Meaning |
|---|---|
| `iterations` | Number of boosting rounds |
| `learning_rate` | Learning rate |
| `depth` | Tree depth |
| `loss_function` | Training objective |
| `eval_metric` | Evaluation metric |
| `verbose` | Training log frequency |

## 5. High-level comparison

| Area | LightGBM | CatBoost | XGBoost |
|---|---|---|---|
| Emphasis | Efficient histogram boosting | Categorical feature handling | General-purpose optimized boosting |
| Common tree growth | Leaf-wise | Symmetric trees by default in common modes; other options exist | Configurable tree growth |
| Categorical workflow | Supported with configuration | Major design strength | Supported with version/configuration requirements |
| Use case | Larger tabular data | Categorical-heavy tabular data | Broad tabular classification/regression |

## 6. Install and verify
```bash
python -m pip install lightgbm catboost xgboost scikit-learn pandas
```

```python
import lightgbm, catboost, xgboost
print(lightgbm.__version__, catboost.__version__, xgboost.__version__)
```

## 7. Compare on numerical tabular data

### 7.1 Prepare dataset
```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X_values, y = make_classification(
    n_samples=1000, n_features=10, n_informative=6,
    n_redundant=2, random_state=42
)
X = pd.DataFrame(X_values, columns=[f"feature_{i}" for i in range(10)])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 7.2 Train candidate models
```python
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

models = {
    "LightGBM": LGBMClassifier(
        n_estimators=150, learning_rate=0.05,
        num_leaves=15, random_state=42, verbosity=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=150, learning_rate=0.05,
        depth=5, verbose=False, random_seed=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150, learning_rate=0.05,
        max_depth=4, eval_metric="logloss", random_state=42
    )
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "trained")
```

### 7.3 Compare metrics
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

rows = []
for name, model in models.items():
    pred = model.predict(X_test)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    })
print(pd.DataFrame(rows).sort_values("f1", ascending=False))
```

This is an initial demonstration. A robust selection requires cross-validation, appropriate preprocessing and runtime measurements.

## 8. CatBoost with categorical features
```python
import pandas as pd
from catboost import CatBoostClassifier

data = pd.DataFrame({
    "contract": ["monthly", "annual", "monthly", "two_year"],
    "region": ["north", "south", "east", "north"],
    "tenure": [3, 24, 5, 30],
    "churn": [1, 0, 1, 0]
})
X_cat = data.drop(columns="churn")
y_cat = data["churn"]

cat_model = CatBoostClassifier(iterations=80, verbose=False, random_seed=42)
cat_model.fit(X_cat, y_cat, cat_features=["contract", "region"])
print(cat_model.predict(X_cat))
```

**Note:** Four rows demonstrate the API only. They cannot establish generalization quality.

## 9. Approved mini-project — Tabular Model Comparison

**Scenario:** An online business wants to predict customer response from mixed numerical and categorical attributes.

**Tasks**
1. Select a classification or regression dataset.
2. Identify numeric and categorical columns.
3. Define target and prediction use case.
4. Create a leakage-safe split.
5. Prepare features using each library's supported interface.
6. Train LightGBM, CatBoost and XGBoost.
7. Compare suitable metrics, training time and model size.
8. Record versions, parameters and limitations.

**Report fields:** Dataset/target, feature types, split method, parameters, metrics, runtime and selection rationale.

**Fairness rules:** Same evaluation records, same target definition, no target leakage, consistent tuning policy and test data used only for final comparison.

## 10. Common mistakes
- Assuming one library is always fastest.
- Comparing defaults as if equally tuned.
- Treating categorical text as meaningful numeric codes.
- Ignoring version-specific APIs.
- Selecting only by accuracy.
- Repeatedly tuning against test data.

## 11. Completion checklist
- [ ] Explain LightGBM histogram and leaf-wise concepts.
- [ ] Explain CatBoost categorical handling.
- [ ] Train and compare three libraries.
- [ ] Explain practical trade-offs.
- [ ] Complete Tabular Model Comparison.

**Next lesson:** Day 57 — Time-series forecasting fundamentals.
