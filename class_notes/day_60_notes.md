# Day 60: Model Pipelines and Preprocessing Integration

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Estimated learning time:** 2–3 hours  
**Approved mini-project:** Reusable ML Pipeline

## 1. Learning objectives
- Explain why preprocessing and model training must be integrated.
- Use ColumnTransformer for mixed numerical and categorical data.
- Build a scikit-learn Pipeline.
- Fit transformations only on training data.
- Evaluate a complete preprocessing-and-model workflow.

## 2. What is an ML pipeline?
An ML pipeline connects ordered data transformations and a final estimator into one workflow.

Example:
Raw input → impute missing values → scale numeric columns / encode categories → train model → predict.

Benefits:
- Same transformations are applied during training and inference.
- Reduces inconsistent manual preprocessing.
- Helps prevent leakage when fitted inside cross-validation.
- Easier to validate, serialize and reuse.

## 3. Why separate preprocessing can be dangerous
Suppose you calculate a mean using the entire dataset before splitting. The test set has influenced the training transformation. This is data leakage.

Instead:
- Split data first.
- Put learned preprocessing steps inside a pipeline.
- Fit the pipeline on training data only.
- Use the fitted pipeline to predict on test or production data.

## 4. Example mixed-type dataset
```python
import pandas as pd

data = pd.DataFrame({
    "age": [22, 35, None, 48, 29, 41, 52, 31, 26, 45],
    "monthly_charge": [30, 80, 45, None, 55, 90, 110, 60, 40, 95],
    "contract": ["monthly", "annual", "monthly", "two_year", "annual",
                 "monthly", "two_year", "annual", "monthly", "two_year"],
    "support_calls": [4, 1, 3, 0, 2, 5, 0, 1, 3, 2],
    "churn": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
})
X = data.drop(columns="churn")
y = data["churn"]
print(X.head())
```

This is small synthetic teaching data, not a realistic performance benchmark.

## 5. Split before fitting transformations
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape)
```

**Explanation:** Stratification preserves class proportions as far as possible. On very small datasets, class counts can constrain the split.

## 6. Build preprocessing with ColumnTransformer
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

numeric_features = ["age", "monthly_charge", "support_calls"]
categorical_features = ["contract"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])
```

**Explanation:**
- Numeric columns receive median imputation and scaling.
- Categorical columns receive most-frequent imputation and one-hot encoding.
- `handle_unknown="ignore"` prevents an unseen category from crashing this encoder at transform time.
- ColumnTransformer applies the correct workflow to each column group.

## 7. Integrate the estimator
```python
from sklearn.linear_model import LogisticRegression

full_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

full_pipeline.fit(X_train, y_train)
print("Pipeline trained")
```

The pipeline learns imputation values, scaling parameters, category levels and model coefficients from the training data.

## 8. Predict and evaluate
```python
from sklearn.metrics import classification_report

predictions = full_pipeline.predict(X_test)
print(classification_report(y_test, predictions, zero_division=0))
```

**Key point:** Pass the original feature columns to the pipeline. It performs the same fitted preprocessing before prediction.

## 9. Cross-validation with a pipeline
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    full_pipeline, X, y, cv=3, scoring="f1"
)
print(scores)
print("Mean F1:", scores.mean())
```

Each fold fits preprocessing using only that fold's training portion. For real datasets, choose a split strategy that respects grouping or time where necessary.

## 10. Inspect pipeline steps
```python
print(full_pipeline.named_steps.keys())
print(full_pipeline.named_steps["model"])
```

Use `named_steps` to access a named component. Avoid manually refitting preprocessing steps outside the pipeline.

## 11. Approved mini-project — Reusable ML Pipeline

**Scenario:** An ML team repeatedly trains churn models and wants one consistent preprocessing-and-model workflow.

**Objective:** Create a reusable pipeline for mixed numerical and categorical data.

### Requirements
1. Identify numerical and categorical columns.
2. Define missing-value strategies.
3. Scale numerical features where suitable.
4. Encode categorical features.
5. Integrate a classifier or regressor.
6. Train and evaluate the complete pipeline.
7. Run cross-validation using the pipeline.
8. Test prediction on a new row containing an unseen category.
9. Document the input schema and learned preprocessing behavior.

### Suggested structure
```text
reusable_ml_pipeline/
├── data/
├── src/
│   ├── preprocessing.py
│   ├── pipeline.py
│   └── evaluate.py
├── tests/
└── README.md
```

### Validation
- Train/test split occurs before fitting.
- Pipeline input columns and data types are documented.
- No target column enters preprocessing.
- Unseen categories are handled intentionally.
- Test metrics are reported only after the pipeline is fitted on training data.
- Pipeline can predict on a raw DataFrame with the expected schema.

## 12. Common mistakes
| Mistake | Better approach |
|---|---|
| Fit scaler before splitting | Put scaler inside pipeline |
| Apply different transforms in production | Reuse the fitted pipeline |
| Forget unseen categories | Configure encoder behavior |
| Pass already transformed data to full pipeline | Pass raw feature columns |
| Use random CV for grouped/time data | Choose group/time-aware validation |
| Save only estimator and not preprocessing | Persist the complete pipeline |

## 13. Lesson completion
- [ ] Explain Pipeline and ColumnTransformer.
- [ ] Integrate numerical and categorical preprocessing.
- [ ] Evaluate with cross-validation.
- [ ] Explain how pipeline fitting prevents leakage.
- [ ] Complete Reusable ML Pipeline.

**Next lesson:** Day 61 — Model selection and experiment tracking.
