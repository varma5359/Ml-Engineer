# Day 63: Scikit-learn Pipelines and Serialization

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Estimated learning time:** 2–3 hours  
**Approved mini-project:** Model Serialization Tool

## 1. Learning objectives
- Build and inspect a scikit-learn Pipeline.
- Explain fit, transform and predict behavior.
- Serialize and reload a fitted pipeline.
- Validate inference input schema.
- Test prediction consistency before and after loading.

## 2. Pipeline recap
A scikit-learn Pipeline chains transformations and a final estimator.

For example:
Input DataFrame → imputer → scaler → logistic regression → predicted class.

The pipeline ensures the same fitted preprocessing is used during inference.

## 3. Build a complete pipeline
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])
pipeline.fit(X_train, y_train)
print("Fitted pipeline")
```

## 4. Understand fit, transform and predict
- `fit(X_train, y_train)`: learns preprocessing and estimator parameters.
- `transform(X)`: applies a learned transformation; the final estimator in a typical pipeline does not need to transform.
- `predict(X)`: runs preprocessing and then predicts.
- `predict_proba(X)`: returns class probability estimates if the estimator supports it.

```python
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)
print(predictions[:5])
print(probabilities[:2])
```

## 5. Inspect pipeline components
```python
print(pipeline.named_steps.keys())
print(pipeline.named_steps["classifier"])
```

Use step names to inspect or configure individual components.

## 6. Serialize using joblib
```python
from pathlib import Path
import joblib

artifact_path = Path("artifacts/churn_pipeline.joblib")
artifact_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, artifact_path)
print("Saved:", artifact_path)
```

## 7. Load in a separate inference workflow
```python
loaded_model = joblib.load(artifact_path)
loaded_predictions = loaded_model.predict(X_test)

print("Loaded model predictions:", loaded_predictions[:5])
```

**Security:** Load only trusted artifacts. joblib uses pickle-based serialization behavior; never load arbitrary files supplied by unknown users.

## 8. Validate prediction consistency
```python
import numpy as np

before = pipeline.predict(X_test)
after = loaded_model.predict(X_test)

print("Identical predictions:", np.array_equal(before, after))
```

**Expected output:** True when the same artifact and compatible environment are used.

Also compare probabilities within a numeric tolerance when required.

## 9. Validate input schema
```python
expected_columns = list(X_train.columns)

def validate_input(frame):
    missing = set(expected_columns) - set(frame.columns)
    extra = set(frame.columns) - set(expected_columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected columns: {sorted(extra)}")

    return frame[expected_columns]

sample = validate_input(X_test.head(2))
print(sample.shape)
```

A production service should also validate data types, allowed ranges, null policy and schema version. This example reorders expected columns but does not perform all those checks.

## 10. Serialization formats
| Format | Common use | Notes |
|---|---|---|
| joblib | Python/scikit-learn objects | Convenient; load only trusted files |
| pickle | Python object serialization | Same untrusted-deserialization risk |
| ONNX | Interoperable model representation for supported estimators | Conversion and operator support must be checked |
| skops.io | Safer scikit-learn-oriented persistence workflow | Review trusted types and compatibility |

Choose based on model support, environment and security needs.

## 11. Approved mini-project — Model Serialization Tool

**Scenario:** A model developer hands a trained estimator to an inference engineer.

**Objective:** Create a tool that saves, loads and validates a model pipeline.

### Requirements
1. Train a scikit-learn pipeline.
2. Save it to an artifact directory.
3. Save metadata containing model name, feature names, target, package versions and training timestamp.
4. Load the artifact in a separate script.
5. Validate input columns and order.
6. Compare predictions before and after serialization.
7. Provide clear errors for invalid input.
8. Document artifact trust and compatibility requirements.

### Suggested structure
```text
model_serialization_tool/
├── artifacts/
├── src/
│   ├── train.py
│   ├── save_model.py
│   ├── load_model.py
│   └── validate_input.py
├── tests/
└── README.md
```

### Validation checklist
- Saved artifact exists.
- Reload succeeds in the documented environment.
- Input schema is validated.
- Prediction shape is correct.
- Predictions match before and after reload.
- No untrusted serialized file is loaded.

## 12. Common mistakes
- Saving only the estimator and omitting preprocessing.
- Changing feature order at inference.
- Loading an artifact from an untrusted source.
- Ignoring scikit-learn version compatibility.
- Treating successful loading as proof of correct predictions.
- Not testing empty or malformed inputs.

## 13. Lesson completion
- [ ] Build a scikit-learn Pipeline.
- [ ] Explain fit and predict.
- [ ] Save and reload the pipeline.
- [ ] Validate schema and predictions.
- [ ] Complete Model Serialization Tool.

**Next lesson:** Day 64 — Model deployment concepts and inference APIs.
