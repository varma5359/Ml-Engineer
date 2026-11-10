# Day 61: Model Selection and Experiment Tracking

**Pathway:** Machine Learning Engineer  
**Phase:** 6 — Advanced ML and Practical Applications  
**Level:** Intermediate  
**Estimated learning time:** 2–3 hours  
**Approved mini-project:** Experiment Comparison Report

## 1. Learning objectives
- Explain model selection and validation.
- Distinguish training, validation and test data.
- Define experiment parameters, metrics and artifacts.
- Track multiple experiments using a consistent record.
- Create a comparison report and justify a model choice.

## 2. What is model selection?
Model selection chooses a model and configuration that meet the problem's requirements and generalize to unseen data.

Consider:
- Predictive performance.
- Generalization and stability.
- Training and inference cost.
- Interpretability and maintainability.
- Data and infrastructure constraints.
- Business impact of different error types.

The highest score on one split is not automatically the best production model.

## 3. Validation design
| Dataset | Purpose |
|---|---|
| Training | Fit model parameters |
| Validation | Tune hyperparameters and select candidates |
| Test | Final evaluation after decisions are complete |

Cross-validation can estimate performance across several splits. The final test set should not become a repeated tuning target.

## 4. What is an experiment?
An experiment is a recorded model run with:
- Run name or ID.
- Dataset/version.
- Code revision.
- Parameters.
- Metrics.
- Artifacts such as plots or model files.
- Environment details.
- Notes and status.

A record makes it possible to reproduce and compare decisions later.

## 5. Track experiments with a simple table
```python
import pandas as pd

experiments = [
    {"run_id": "run_001", "model": "LogisticRegression",
     "C": 1.0, "f1": 0.78, "train_seconds": 0.4},
    {"run_id": "run_002", "model": "RandomForest",
     "C": None, "f1": 0.82, "train_seconds": 2.1},
    {"run_id": "run_003", "model": "GradientBoosting",
     "C": None, "f1": 0.84, "train_seconds": 3.8}
]

experiment_table = pd.DataFrame(experiments)
print(experiment_table.sort_values("f1", ascending=False))
```

**Explanation:** This is a manually maintained experiment log. It demonstrates the fields to record, not a substitute for tracking dataset and code versions.

## 6. Track runs using MLflow

### 6.1 Install and import
```bash
python -m pip install mlflow scikit-learn pandas
```

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
```

### 6.2 Load data and split
```python
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 6.3 Start an experiment and log a run
```python
mlflow.set_experiment("day_61_model_selection")

with mlflow.start_run(run_name="random_forest_baseline"):
    n_estimators = 150
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
    mlflow.log_metric("f1", f1_score(y_test, predictions))
    mlflow.sklearn.log_model(model, name="model")

print("Run logged")
```

**Expected output:** A run is logged and a model artifact is recorded. MLflow may print a local tracking URI or artifact information depending on version.

**Important:** For a real model-selection process, log validation metrics during tuning and reserve a separate test set for final evaluation. The compact demonstration logs test metrics only to show API usage; do not repeat this pattern while tuning candidates.

### 6.4 Inspect the experiment
```python
runs = mlflow.search_runs(
    experiment_names=["day_61_model_selection"],
    order_by=["metrics.f1 DESC"]
)
print(runs[["run_name", "params.n_estimators", "params.max_depth", "metrics.f1"]])
```

The returned columns depend on the logged parameters and MLflow version.

## 7. Compare experiments responsibly
- Compare the same metric and same validation policy.
- Record data and code versions.
- Consider mean and variation across folds.
- Avoid comparing metrics from different test populations.
- Track runtime and model size where important.
- Explain why a selected model meets the business requirement.

## 8. Approved mini-project — Experiment Comparison Report

**Scenario:** An ML team has trained multiple candidate models and needs a traceable selection decision.

**Objective:** Run and document multiple candidate experiments.

### Requirements
1. Define task, target and primary metric.
2. Create train/validation/test strategy.
3. Train at least three candidates or configurations.
4. Record parameters, metrics and run names.
5. Track experiments with MLflow or a structured CSV/JSON log.
6. Record dataset and code revision identifiers.
7. Compare generalization, runtime and complexity.
8. Select a candidate and write a short decision report.

### Suggested report fields
| Field | Example |
|---|---|
| Run ID | Unique run name |
| Dataset version | File checksum or DVC revision |
| Code revision | Git commit |
| Parameters | Estimators, depth, learning rate |
| Validation metric | F1, RMSE, etc. |
| Runtime | Seconds |
| Artifact | Model or plot path |
| Decision | Keep, reject or investigate |

## 9. Common mistakes
- Tuning on the test set.
- Logging metrics without data/version details.
- Comparing runs with different target definitions.
- Selecting only by one metric.
- Overwriting artifacts with ambiguous names.
- Treating a run record as proof that the experiment is reproducible.

## 10. Lesson completion
- [ ] Explain training, validation and test roles.
- [ ] Define experiment metadata.
- [ ] Log parameters, metrics and a model with MLflow.
- [ ] Compare candidates fairly.
- [ ] Complete Experiment Comparison Report.

**Next lesson:** Day 62 — ML project reproducibility and artifact management.
